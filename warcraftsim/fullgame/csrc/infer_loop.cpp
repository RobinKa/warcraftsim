// The inference server's loop in C++ (--infer-native): a thread of the learner's process that answers the
// games' network calls without Python's GIL. Python captures every call's CUDA graph first (fused kernels,
// the server's own random generator) and hands over pointers: the graphs, their input and output tensors,
// pinned host buffers, the shared rows (SharedRows) and each past-snapshot slot's weights. The loop:
// read the games' binary requests from their pipes, gather their rows into a call's input, copy a past
// snapshot's weights into its slot when it changes, replay the graphs on high-priority streams (the next
// round launched before the last one is answered), write the answers into the shared rows and reply.
// Python is called (with the GIL) only to reload the current network, to load a new past snapshot, and
// for the statistics.
//
// A request (multiprocessing's send_bytes: a 4-byte big-endian length, then):
//   u32 request id, u8 count, then count x {u16 entities, u16 own units, u32 row, u16 key length, key}
// A reply: u32 request id, u8 count, then count x {i32 version, u16 O, u16 memory width}
// (version -2: no answer, the call failed).
#include <torch/extension.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <poll.h>
#include <unistd.h>
#include <arpa/inet.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "past_hold.h"

namespace {

struct Shape {
    int B = 0, E = 0, W = 0, cols = 0;
    at::cuda::CUDAGraph* graph = nullptr;
    float* x_dev = nullptr;
    float* out_dev = nullptr;
    // pinned host buffers: [bank][k] (k: the k-th call of this shape in a round; a bank per round in flight)
    static constexpr int BANKS = 4;
    std::vector<float*> hin[BANKS], hout[BANKS];
};

struct Group {  // the current network, or one past-snapshot slot of one architecture
    std::vector<Shape> shapes;
    int max_batch = 0;
    std::vector<int> buckets, widths;
    std::vector<std::pair<void*, size_t>> dst;  // a slot's weights and buffers (past only)
    std::string holding;                        // the snapshot in the slot
    int arch = -1;
    std::unique_ptr<c10::cuda::CUDAStream> stream;
    // a round's events on this stream, one per bank, made once: creating and destroying events takes the CUDA
    // context's lock exclusively, and next to the learner's calls that waited ~4 ms a round
    cudaEvent_t start_ev[Shape::BANKS] = {}, end_ev[Shape::BANKS] = {};
};

cudaEvent_t bank_event(cudaEvent_t& ev, unsigned flags) {
    if (!ev) C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev, flags));
    return ev;
}

struct Snapshot {
    int arch = -1;
    std::vector<std::pair<const void*, size_t>> src;
};

struct Request {  // a game's request: the answers, and how many are still to come (a past snapshot's may be held
    int conn = -1;  // back beyond the round that read it: --infer-past-period-ms)
    uint32_t rid = 0;
    std::vector<std::array<int32_t, 3>> answers;
    int left = 0;
    bool sent = false;
};

struct Item {
    int conn, k, n, n_own, row;
    std::shared_ptr<Request> request;
};

struct Call {
    Group* group;
    Shape* shape;
    float* hout;
    std::vector<Item> items;
    int version;
};

struct Pending {
    std::vector<Call> calls;
    std::vector<cudaEvent_t> events;
    std::vector<cudaEvent_t> starts;  // (timed: each stream's part of the round on the GPU, start to end)
    int learner_busy = 0;  // (the learner was in its update when the round was launched)
    double t_launched = 0;  // (the round's last call enqueued)
    std::vector<std::shared_ptr<Request>> requests;  // read this round, or with items launched in it (in the order read)
    double t0;
};

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool read_full(int fd, void* buf, size_t n) {
    auto* p = static_cast<char*>(buf);
    while (n) {
        ssize_t r = ::read(fd, p, n);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) continue;
            return false;
        }
        p += r;
        n -= r;
    }
    return true;
}

bool write_full(int fd, const void* buf, size_t n) {
    auto* p = static_cast<const char*>(buf);
    while (n) {
        ssize_t r = ::write(fd, p, n);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) continue;
            return false;
        }
        p += r;
        n -= r;
    }
    return true;
}

class Loop {
   public:
    // the shared rows' arrays (float32), as SharedRows lays them out
    float *ent = nullptr, *typ = nullptr, *cur = nullptr, *glob = nullptr, *n_own_rows = nullptr, *avail = nullptr,
          *h = nullptr, *out = nullptr;
    int E_max = 0, F = 0, O_max = 0, G = 0, NO = 0, d = 0, hw = 1, out_w = 0;
    std::vector<int> fds;
    Group live;
    std::vector<std::vector<Group>> past;  // [arch][slot]
    int past_max_batch = 16;
    double period = 0.004;
    double past_period = 0;  // (--infer-past-period-ms: a past network's calls at most this often, see PastHold)
    int version = -1;
    int learner_busy = 0;  // (set by the learner around its updates: the statistics keep its rounds apart)
    py::object reload_fn, snapshot_fn, stats_fn;

    void start() {
        stop_ = false;
        thread_ = std::thread([this] { run(); });
    }
    void stop() {
        stop_ = true;
        if (thread_.joinable()) {
            py::gil_scoped_release nogil;
            thread_.join();
        }
    }
    bool failed() const { return failed_; }
    std::string error() const { return error_; }

   private:
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> failed_{false};
    std::string error_;
    std::unordered_map<std::string, Snapshot> snapshots_;
    PastHold<Item> hold_;
    // statistics since the last report
    double st_t = 0, st_busy = 0, st_wait = 0, st_idle = 0;
    double st_lat[2] = {0, 0}, st_gpu[2] = {0, 0};  // rounds' latency and GPU time, the learner idle / updating
    double st_launch[2] = {0, 0}, st_tail[2] = {0, 0};  // of the latency: enqueueing the round; answering it
    long st_n[2] = {0, 0};
    long st_rounds = 0, st_calls = 0, st_past = 0, st_rows = 0, st_ents = 0, st_ents_max = 0;
    long st_past_rows = 0, st_swaps = 0;  // (a past network's weights copied into a slot)
    // calls by their widest row (8 entities a bin) and their rows (8 a bin), [0] the current network's, [1] past ones':
    // what the padding of --infer-entities and the batch sizes cost, from the run itself
    static constexpr int WIDE_BINS = 20, ROWS_BINS = 8;
    std::array<std::array<long, WIDE_BINS>, 2> st_wide{};
    std::array<std::array<long, ROWS_BINS>, 2> st_nrows{};

    float* host(std::vector<float*>& pool, size_t k, size_t n) {
        while (pool.size() <= k) {
            float* p = nullptr;
            C10_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&p), n * sizeof(float), cudaHostAllocDefault));
            std::fill(p, p + n, 0.0f);  // (not zeroed by CUDA: may hold what a freed pinned block held)
            pool.push_back(p);
        }
        return pool[k];
    }

    Shape* pick(Group& g, int rows, int widest) {
        int B = -1, E = -1;
        for (int b : g.buckets)
            if (rows <= b) { B = b; break; }
        for (int e : g.widths)
            if (widest <= e) { E = e; break; }
        if (B < 0 || E < 0) return nullptr;
        for (auto& s : g.shapes)
            if (s.B == B && s.E == E) return &s;
        return nullptr;
    }

    // Only the rows asked for are written (and copied to the GPU, and back): the call's other rows keep what
    // an earlier call left there, valid inputs whose answers nobody reads (padding the whole batch with
    // zeros was ~1 MB of writing and copying a call, for ~13 rows). Within a row, the entities past its own
    // are masked out, so only the mask is written to the row's end.
    void gather(Shape& s, float* hin, const std::vector<Item>& items) {
        const int E = s.E, W = s.W;
        const int a_ent = 0, a_glob = E * F, a_typ = a_glob + G, a_cur = a_typ + E, a_mask = a_cur + E, a_own = a_mask + E,
                  a_avail = a_own + 1, a_h = a_avail + NO;
        for (size_t i = 0; i < items.size(); i++) {
            const Item& it = items[i];
            float* r = hin + i * W;
            const int n = std::max(1, std::min(it.n, E));
            const size_t row = it.row;
            // the row's entities, then zeros up to the call's E: the buffer is pinned memory as cudaHostAlloc gave it
            // or as an earlier call left it, and masked entities still go through the type and order embeddings
            // (a stale unit type past the vocabulary failed a device-side assert in the fused call)
            std::memcpy(r + a_ent, ent + row * E_max * F, sizeof(float) * n * F);
            std::fill(r + a_ent + size_t(n) * F, r + a_ent + size_t(E) * F, 0.0f);
            std::memcpy(r + a_glob, glob + row * G, sizeof(float) * G);
            std::memcpy(r + a_typ, typ + row * E_max, sizeof(float) * n);
            std::fill(r + a_typ + n, r + a_typ + E, 0.0f);
            std::memcpy(r + a_cur, cur + row * E_max, sizeof(float) * n);
            std::fill(r + a_cur + n, r + a_cur + E, 0.0f);
            std::fill(r + a_mask, r + a_mask + n, 1.0f);
            std::fill(r + a_mask + n, r + a_mask + E, 0.0f);
            r[a_own] = std::min(n_own_rows[row], float(O_max));
            std::memcpy(r + a_avail, avail + row * NO, sizeof(float) * NO);
            if (d) std::memcpy(r + a_h, h + row * hw, sizeof(float) * d);
        }
    }

    Snapshot* snapshot(const std::string& key) {
        auto it = snapshots_.find(key);
        if (it != snapshots_.end()) return &it->second;
        py::gil_scoped_acquire gil;  // (a new league member: rare)
        py::tuple t = snapshot_fn(key);
        Snapshot s;
        s.arch = t[0].cast<int>();
        for (auto p : t[1].cast<py::list>()) {
            auto pr = p.cast<py::tuple>();
            s.src.emplace_back(reinterpret_cast<const void*>(pr[0].cast<uintptr_t>()), pr[1].cast<size_t>());
        }
        return &(snapshots_[key] = std::move(s));
    }

    void reload() {
        c10::cuda::CUDAStreamGuard guard(*live.stream);  // (the weights copied in order with the calls)
        py::gil_scoped_acquire gil;
        version = reload_fn().cast<int>();
    }

    void report() {
        py::gil_scoped_acquire gil;
        py::dict s;
        s["dt"] = now() - st_t;
        s["rounds"] = st_rounds;
        s["calls"] = st_calls;
        s["past"] = st_past;
        s["rows"] = st_rows;
        s["ents"] = st_ents;
        s["ents_max"] = st_ents_max;
        s["busy"] = st_busy;
        s["wait"] = st_wait;
        s["lat"] = py::make_tuple(st_lat[0], st_lat[1]);
        s["gpu"] = py::make_tuple(st_gpu[0], st_gpu[1]);
        s["n"] = py::make_tuple(st_n[0], st_n[1]);
        s["launch"] = py::make_tuple(st_launch[0], st_launch[1]);
        s["tail"] = py::make_tuple(st_tail[0], st_tail[1]);
        s["idle"] = st_idle;
        s["past_rows"] = st_past_rows;
        s["swaps"] = st_swaps;
        s["held_rows"] = hold_.rows_out;
        s["held_wait"] = hold_.waited;
        s["wide"] = st_wide;
        s["nrows"] = st_nrows;
        stats_fn(s);
        st_t = now();
        st_rounds = st_calls = st_past = st_rows = st_ents = st_ents_max = 0;
        st_past_rows = st_swaps = 0;
        hold_.rows_out = 0, hold_.waited = 0;
        st_wide = {};
        st_nrows = {};
        st_busy = st_wait = st_idle = 0;
        st_lat[0] = st_lat[1] = st_gpu[0] = st_gpu[1] = 0;
        st_n[0] = st_n[1] = 0;
        st_launch[0] = st_launch[1] = st_tail[0] = st_tail[1] = 0;
    }

    void complete(Pending& p) {
        double t = now();
        const double t_complete = t;
        float gpu = 0;
        for (size_t i = 0; i < p.events.size(); i++) {
            C10_CUDA_CHECK(cudaEventSynchronize(p.events[i]));
            float ms = 0;
            if (i < p.starts.size()) C10_CUDA_CHECK(cudaEventElapsedTime(&ms, p.starts[i], p.events[i]));
            gpu = std::max(gpu, ms);  // (the events are the groups' own, kept for the bank's next round)
        }
        st_wait += now() - t;
        st_gpu[p.learner_busy] += gpu / 1000.0;
        for (auto& c : p.calls) {
            const int cols = c.shape->cols, O = std::min(O_max, c.shape->E);
            const int width = d ? cols - 5 * O - 2 : 0;
            for (size_t i = 0; i < c.items.size(); i++) {
                const Item& it = c.items[i];
                std::memcpy(out + size_t(it.row) * out_w, c.hout + i * cols, sizeof(float) * cols);
                it.request->answers[it.k] = {c.version, O, width};
                it.request->left--;  // (here, not at the launch: an earlier round's completion must not answer a
            }                        // request whose held row is on the GPU in a later one)
        }
        for (auto& r : p.requests) {  // (those all answered; the rest go with the round of their last item)
            if (r->left > 0 || r->sent) continue;
            r->sent = true;
            const int conn = r->conn;
            std::vector<char> buf(4 + 4 + 1 + r->answers.size() * 8);
            uint32_t len = htonl(uint32_t(buf.size() - 4));
            std::memcpy(buf.data(), &len, 4);
            std::memcpy(buf.data() + 4, &r->rid, 4);
            buf[8] = char(r->answers.size());
            char* q = buf.data() + 9;
            for (auto& a : r->answers) {
                int32_t v = a[0];
                uint16_t o = uint16_t(a[1]), w = uint16_t(a[2]);
                std::memcpy(q, &v, 4);
                std::memcpy(q + 4, &o, 2);
                std::memcpy(q + 6, &w, 2);
                q += 8;
            }
            if (fds[conn] >= 0 && !write_full(fds[conn], buf.data(), buf.size())) fds[conn] = -1;
        }
        st_rounds++;
        st_busy += now() - p.t0;
        st_lat[p.learner_busy] += now() - p.t0;
        st_launch[p.learner_busy] += p.t_launched - p.t0;
        st_tail[p.learner_busy] += now() - t_complete;
        st_n[p.learner_busy]++;
    }

    void run() {
        try {
            loop();
        } catch (const std::exception& e) {
            error_ = e.what();
            failed_ = true;
        }
    }

    bool done(const Pending& p) {
        for (auto ev : p.events) {
            cudaError_t e = cudaEventQuery(ev);
            if (e == cudaErrorNotReady) return false;
            C10_CUDA_CHECK(e);
        }
        return true;
    }

    // Event-driven: rounds in flight (up to BANKS - 1) are answered as their GPU work finishes, and new
    // requests are read and launched meanwhile (one round in flight, answered before the next was read,
    // left the loop waiting for the GPU half of the time).
    void loop() {
        auto device_guard = c10::cuda::CUDAGuard(live.stream->device_index());
        std::vector<pollfd> pfds;
        std::deque<std::unique_ptr<Pending>> inflight;
        int bank = 0;
        double t_round = 0, t_reload = now(), t_report = now();
        st_t = now();
        hold_.period = past_period;
        hold_.full = size_t(past_max_batch);
        while (!stop_) {
            if (now() - t_reload > 5.0) {
                reload();
                t_reload = now();
            }
            if (now() - t_report > 10.0 && st_rounds) {
                report();
                t_report = now();
            }
            while (!inflight.empty() && done(*inflight.front())) {  // (in order: a game has one request at a time)
                complete(*inflight.front());
                inflight.pop_front();
            }
            pfds.clear();
            for (int fd : fds) pfds.push_back({fd, short(fd >= 0 ? POLLIN : 0), 0});
            double t_idle = now();
            const bool full = int(inflight.size()) >= Shape::BANKS - 1;
            int timeout = inflight.empty() ? 100 : 0;
            if (timeout && hold_.on() && hold_.holding())  // (held rows' deadline: nothing else may come to wake the loop)
                timeout = int(std::clamp(std::ceil((hold_.next_due() - now()) * 1000), 0.0, 100.0));
            int ready = full ? 0 : ::poll(pfds.data(), pfds.size(), timeout);
            const bool due = !full && hold_.on() && hold_.holding() && now() >= hold_.next_due();
            if (ready <= 0 && due)  // (a round for the held rows alone)
                for (auto& p : pfds) p.revents = 0;
            if (ready <= 0 && !due) {
                if (!inflight.empty()) {  // nothing new: wait a little for the GPU (or the oldest round, if full)
                    if (full) {
                        double t = now();
                        complete(*inflight.front());
                        inflight.pop_front();
                        st_wait += now() - t;
                    } else {
                        ::poll(pfds.data(), pfds.size(), 0);
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                    }
                } else {
                    st_idle += now() - t_idle;
                }
                continue;
            }
            // a round at most every `period` seconds
            double pause = t_round + period - now();
            if (pause > 0) {
                std::this_thread::sleep_for(std::chrono::duration<double>(pause));
                for (auto& p : pfds) p.revents = 0;
                ::poll(pfds.data(), pfds.size(), 0);
            }
            const double t_last = t_round;
            t_round = now();
            auto round = std::make_unique<Pending>();
            round->t0 = t_round;
            round->learner_busy = learner_busy ? 1 : 0;
            // the requests: per network key, its items in order
            std::map<std::string, std::vector<Item>> groups;
            for (size_t c = 0; c < pfds.size(); c++) {
                if (!(pfds[c].revents & (POLLIN | POLLHUP | POLLERR)) || fds[c] < 0) continue;
                uint32_t len_be;
                if (!read_full(fds[c], &len_be, 4)) { fds[c] = -1; continue; }
                std::vector<char> msg(ntohl(len_be));
                if (!read_full(fds[c], msg.data(), msg.size())) { fds[c] = -1; continue; }
                uint32_t rid;
                std::memcpy(&rid, msg.data(), 4);
                int count = uint8_t(msg[4]);
                auto rep = std::make_shared<Request>();
                rep->conn = int(c), rep->rid = rid, rep->left = count;
                rep->answers.assign(count, {-2, 0, 0});
                round->requests.push_back(rep);
                const char* q = msg.data() + 5;
                for (int k = 0; k < count; k++) {
                    uint16_t n, own, klen;
                    uint32_t row;
                    std::memcpy(&n, q, 2);
                    std::memcpy(&own, q + 2, 2);
                    std::memcpy(&row, q + 4, 4);
                    std::memcpy(&klen, q + 8, 2);
                    std::string key(q + 10, klen);
                    q += 10 + klen;
                    groups[key].push_back({int(c), k, int(n), int(own), int(row), rep});
                }
            }
            if (hold_.on()) {  // past snapshots' rows wait their turn (a request's current row goes now: same answer)
                for (auto it = groups.begin(); it != groups.end();) {
                    if (it->first == "current") { ++it; continue; }
                    hold_.add(it->first, std::move(it->second), t_round);
                    it = groups.erase(it);
                }
                hold_.take_due(now(), groups, [&](const Item& it) { round->requests.push_back(it.request); });
                if (groups.empty() && std::none_of(round->requests.begin(), round->requests.end(),
                                                   [](auto& r) { return r->left == 0; })) {
                    t_round = t_last;  // (all held: no round, and the pace not spent)
                    continue;
                }
            }
            // the calls: the current network, then each past snapshot in a slot of its architecture
            std::vector<Group*> used;
            std::vector<std::pair<Group*, int>> taken;  // (slot group, how often taken this round)
            std::map<Shape*, int> nth;
            for (auto& [key, items] : groups) {
                Group* g = nullptr;
                Snapshot* snap = nullptr;
                if (key == "current") {
                    g = &live;
                } else {
                    snap = snapshot(key);
                    auto& slots = past.at(snap->arch);
                    Group* held = nullptr; Group* empty = nullptr; Group* free_ = nullptr;
                    for (auto& s : slots) {
                        bool busy = std::any_of(taken.begin(), taken.end(), [&](auto& t) { return t.first == &s; });
                        if (busy) continue;
                        if (s.holding == key && !held) held = &s;
                        if (s.holding.empty() && !empty) empty = &s;
                        if (!free_) free_ = &s;
                    }
                    g = held ? held : empty ? empty : free_ ? free_ : &slots[taken.size() % slots.size()];
                    taken.push_back({g, 1});
                }
                const bool first = std::find(used.begin(), used.end(), g) == used.end();
                if (first) used.push_back(g);
                c10::cuda::CUDAStreamGuard guard(*g->stream);
                if (first) {  // (the stream's part of the round starts here: in `used` order, as the end events)
                    cudaEvent_t ev = bank_event(g->start_ev[bank], cudaEventDefault);
                    C10_CUDA_CHECK(cudaEventRecord(ev, g->stream->stream()));
                    round->starts.push_back(ev);
                }
                if (snap && g->holding != key) {  // the snapshot's weights into the slot (in order on its stream)
                    for (size_t i = 0; i < g->dst.size() && i < snap->src.size(); i++)
                        C10_CUDA_CHECK(cudaMemcpyAsync(g->dst[i].first, snap->src[i].first, g->dst[i].second,
                                                       cudaMemcpyDeviceToDevice, g->stream->stream()));
                    g->holding = key;
                    st_swaps++;
                }
                const int size = g->max_batch;
                for (size_t a = 0; a < items.size(); a += size) {
                    std::vector<Item> part(items.begin() + a, items.begin() + std::min(items.size(), a + size));
                    int widest = 1;
                    for (auto& it : part) widest = std::max(widest, it.n);
                    Shape* s = pick(*g, int(part.size()), widest);
                    if (!s) {  // (no graph for it: no answer, version -2)
                        for (auto& it : part) it.request->left--;
                        continue;
                    }
                    int k = nth[s]++;
                    float* hin = host(s->hin[bank], k, size_t(s->B) * s->W);
                    float* hout = host(s->hout[bank], k, size_t(s->B) * s->cols);
                    gather(*s, hin, part);
                    const size_t used = part.size();
                    C10_CUDA_CHECK(cudaMemcpyAsync(s->x_dev, hin, sizeof(float) * used * s->W, cudaMemcpyHostToDevice,
                                                   g->stream->stream()));
                    s->graph->replay();
                    C10_CUDA_CHECK(cudaMemcpyAsync(hout, s->out_dev, sizeof(float) * used * s->cols,
                                                   cudaMemcpyDeviceToHost, g->stream->stream()));
                    round->calls.push_back({g, s, hout, std::move(part), g == &live ? version : -1});
                    st_calls++;
                    st_past += g != &live;
                    st_rows += round->calls.back().items.size();
                    if (g != &live) st_past_rows += round->calls.back().items.size();
                    for (auto& it : round->calls.back().items) st_ents += it.n;
                    st_ents_max += widest;
                    st_wide[g != &live][std::min(WIDE_BINS - 1, (widest - 1) / 8)]++;
                    st_nrows[g != &live][std::min(ROWS_BINS - 1, (int(round->calls.back().items.size()) - 1) / 8)]++;
                }
            }
            for (Group* g : used) {
                cudaEvent_t ev = bank_event(g->end_ev[bank], cudaEventBlockingSync);  // (timed: see starts)
                C10_CUDA_CHECK(cudaEventRecord(ev, g->stream->stream()));
                round->events.push_back(ev);
            }
            round->t_launched = now();
            inflight.push_back(std::move(round));
            bank = (bank + 1) % Shape::BANKS;
        }
        while (!inflight.empty()) {
            complete(*inflight.front());
            inflight.pop_front();
        }
    }
};

void add_shape(Group& g, int B, int E, int W, int cols, py::object graph, uintptr_t x, uintptr_t out) {
    Shape s;
    s.B = B, s.E = E, s.W = W, s.cols = cols;
    s.graph = graph.cast<at::cuda::CUDAGraph*>();
    s.x_dev = reinterpret_cast<float*>(x);
    s.out_dev = reinterpret_cast<float*>(out);
    g.shapes.push_back(std::move(s));
}

// a group's stream: one made by the caller (a stream in the inference loop's part of the GPU: gpu_split.py), else a
// high-priority one from torch's pool
c10::cuda::CUDAStream make_stream(uintptr_t stream) {
    if (stream) return c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream), c10::cuda::current_device());
    return c10::cuda::getStreamFromPool(true);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Loop>(m, "Loop")
        .def(py::init<>())
        .def("set_rows", [](Loop& l, std::vector<uintptr_t> p, int E_max, int F, int O_max, int G, int NO, int d) {
            l.ent = reinterpret_cast<float*>(p[0]); l.typ = reinterpret_cast<float*>(p[1]);
            l.cur = reinterpret_cast<float*>(p[2]); l.glob = reinterpret_cast<float*>(p[3]);
            l.n_own_rows = reinterpret_cast<float*>(p[4]); l.avail = reinterpret_cast<float*>(p[5]);
            l.h = reinterpret_cast<float*>(p[6]); l.out = reinterpret_cast<float*>(p[7]);
            l.E_max = E_max, l.F = F, l.O_max = O_max, l.G = G, l.NO = NO, l.d = d, l.hw = std::max(d, 1);
            l.out_w = 5 * O_max + 2 + std::max(d, 1);
        })
        .def("set_fds", [](Loop& l, std::vector<int> fds) { l.fds = fds; })
        .def("set_live", [](Loop& l, int max_batch, std::vector<int> buckets, std::vector<int> widths, uintptr_t stream) {
            l.live.max_batch = max_batch, l.live.buckets = buckets, l.live.widths = widths;
            l.live.stream = std::make_unique<c10::cuda::CUDAStream>(make_stream(stream));
        }, py::arg("max_batch"), py::arg("buckets"), py::arg("widths"), py::arg("stream") = 0)
        .def("add_live_shape", [](Loop& l, int B, int E, int W, int cols, py::object g, uintptr_t x, uintptr_t o) {
            add_shape(l.live, B, E, W, cols, g, x, o);
        })
        .def("add_slot", [](Loop& l, int arch, int max_batch, std::vector<int> buckets, std::vector<int> widths,
                            std::vector<std::pair<uintptr_t, size_t>> dst, uintptr_t stream) {
            if (int(l.past.size()) <= arch) l.past.resize(arch + 1);
            l.past[arch].emplace_back();
            Group& g = l.past[arch].back();
            g.arch = arch, g.max_batch = max_batch, g.buckets = buckets, g.widths = widths;
            for (auto& [p, n] : dst) g.dst.emplace_back(reinterpret_cast<void*>(p), n);
            g.stream = std::make_unique<c10::cuda::CUDAStream>(make_stream(stream));
            return int(l.past[arch].size()) - 1;
        }, py::arg("arch"), py::arg("max_batch"), py::arg("buckets"), py::arg("widths"), py::arg("dst"), py::arg("stream") = 0)
        .def("add_slot_shape", [](Loop& l, int arch, int slot, int B, int E, int W, int cols, py::object g, uintptr_t x,
                                  uintptr_t o) { add_shape(l.past.at(arch).at(slot), B, E, W, cols, g, x, o); })
        .def("set_callbacks", [](Loop& l, py::object reload, py::object snapshot, py::object stats) {
            l.reload_fn = reload, l.snapshot_fn = snapshot, l.stats_fn = stats;
        })
        .def_readwrite("period", &Loop::period)
        .def_readwrite("past_period", &Loop::past_period)
        .def_readwrite("version", &Loop::version)
        .def_readwrite("learner_busy", &Loop::learner_busy)
        .def("start", &Loop::start)
        .def("stop", &Loop::stop)
        .def("failed", &Loop::failed)
        .def("error", &Loop::error);
}
