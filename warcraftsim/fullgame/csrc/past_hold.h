// --infer-past-period-ms: the inference loop's past-snapshot rows held back so that they batch up. A call costs about
// the same on the GPU whatever its rows (~1 ms: ~3 rows a past call against ~19 a current one), and a game against a
// past snapshot is one of few, so each network's rows come in ones and twos, each a call of their own.
//
// The rule: a network's calls at most once every `period` seconds, counted from its last call. Rows arriving after a
// quiet spell go at once (a wait from each network's first row would cost the same calls and add a wait to all of
// them); a network with a full call's rows (`full`, the largest batch a past call has) goes at once too. So a row waits
// at most `period`: the loop wakes for next_due() whatever else arrives (a game waits for its answer alone, and every
// held row has its deadline, so nothing here can wait for anything else).
//
// No GPU here, no torch: tests/unit/test_past_hold.py compiles it with g++ and plays arrivals through it.
#pragma once

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

template <class Item>
class PastHold {
   public:
    double period = 0;  // seconds (0: nothing held)
    size_t full = 16;
    // since the last read_stats(): rows that left the hold, and the seconds they waited in all
    long rows_out = 0;
    double waited = 0;

    bool on() const { return period > 0; }

    void add(const std::string& key, std::vector<Item>&& items, double t) {
        Net& n = nets_[key];
        for (auto& it : items) {
            n.items.push_back(std::move(it));
            n.since.push_back(t);
        }
    }

    // when the earliest network holding rows may go (infinity: nothing held)
    double next_due() const {
        double t = std::numeric_limits<double>::infinity();
        for (auto& [key, n] : nets_)
            if (!n.items.empty()) t = std::min(t, n.items.size() >= full ? n.last : n.last + period);
        return t;
    }

    bool holding() const {
        for (auto& [key, n] : nets_)
            if (!n.items.empty()) return true;
        return false;
    }

    // the networks due at time t: their rows into groups[key] (after any there), `each` called on every row. A network
    // whose rows have all gone and whose period has passed is forgotten: as good as never called (and the keys of a
    // long run's snapshots don't pile up here)
    template <class Map, class Each>
    void take_due(double t, Map& groups, Each each) {
        for (auto it = nets_.begin(); it != nets_.end();) {
            Net& n = it->second;
            const bool due = t >= n.last + period;
            if (!n.items.empty() && (due || n.items.size() >= full)) {
                auto& out = groups[it->first];
                for (size_t i = 0; i < n.items.size(); i++) {
                    waited += t - n.since[i];
                    each(n.items[i]);
                    out.push_back(std::move(n.items[i]));
                }
                rows_out += long(n.items.size());
                n.items.clear();
                n.since.clear();
                n.last = t;
                ++it;
            } else if (n.items.empty() && due) {
                it = nets_.erase(it);
            } else {
                ++it;
            }
        }
    }

    size_t networks() const { return nets_.size(); }

   private:
    struct Net {
        std::vector<Item> items;
        std::vector<double> since;
        double last = -std::numeric_limits<double>::infinity();
    };
    std::map<std::string, Net> nets_;
};
