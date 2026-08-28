#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

constexpr int PLAYERS = 3;
constexpr int ACTIONS = 9;
constexpr double EPS = 1e-9;
constexpr int FOLD = 0, CHECK = 1, CALL = 2, MIN_RAISE = 3;
constexpr int RAISE_2X = 4, RAISE_3X = 5, HALF_POT = 6, POT = 7, ALL_IN = 8;
constexpr int PREFLOP = 0, FLOP = 1, TURN = 2, RIVER = 3;
const std::array<std::string, ACTIONS> ACTION_NAMES = {
    "fold", "check", "call", "min_raise", "raise_2x", "raise_3x",
    "half_pot", "pot", "all_in"
};

struct ActionRecord {
    int player = 0;
    int street = 0;
    int action = 0;
    std::string action_name;
    double amount = 0;
    double contribution_after = 0;
    double current_bet_before = 0;
    double current_bet_after = 0;
    double pot_after = 0;
    bool full_raise = false;
};

struct SidePot {
    double amount = 0;
    double cap = 0;
    std::vector<int> contributors;
    std::vector<int> eligible;
};

template <typename T, std::size_t Capacity>
class FixedVector {
public:
    FixedVector() = default;
    explicit FixedVector(const std::vector<T>& values) { assign(values); }

    void assign(const std::vector<T>& values) {
        if (values.size() > Capacity) throw py::value_error("packed field exceeds capacity");
        size_ = values.size();
        std::copy(values.begin(), values.end(), data_.begin());
    }
    FixedVector& operator=(const std::vector<T>& values) { assign(values); return *this; }
    void resize(std::size_t size) {
        if (size > Capacity) throw py::value_error("packed field exceeds capacity");
        size_ = size;
    }
    void push_back(const T& value) {
        if (size_ >= Capacity) throw std::runtime_error("packed field capacity exceeded");
        data_[size_++] = value;
    }
    void pop_back() {
        if (!size_) throw std::runtime_error("cannot pop an empty packed field");
        --size_;
    }
    T& back() { if (!size_) throw std::runtime_error("empty packed field"); return data_[size_ - 1]; }
    const T& back() const { if (!size_) throw std::runtime_error("empty packed field"); return data_[size_ - 1]; }
    bool empty() const { return size_ == 0; }
    std::size_t size() const { return size_; }
    T& operator[](std::size_t index) { return data_[index]; }
    const T& operator[](std::size_t index) const { return data_[index]; }
    auto begin() { return data_.begin(); }
    auto end() { return data_.begin() + static_cast<std::ptrdiff_t>(size_); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.begin() + static_cast<std::ptrdiff_t>(size_); }
    std::vector<T> vector() const { return std::vector<T>(begin(), end()); }

private:
    std::array<T, Capacity> data_{};
    std::size_t size_ = 0;
};

struct HistoryNode {
    ActionRecord event;
    std::shared_ptr<const HistoryNode> previous;
};

struct State {
    FixedVector<int, 52> deck;
    FixedVector<int, 5> board;
    FixedVector<int, 3> burned;
    std::array<FixedVector<int, 2>, PLAYERS> hole;
    std::array<double, PLAYERS> stacks{}, initial_stacks{}, total_contrib{}, street_contrib{};
    std::array<bool, PLAYERS> folded{}, all_in{}, raise_rights{}, alive{}, eliminated{};
    std::array<double, PLAYERS> last_action_bet{};
    std::array<bool, PLAYERS> has_last_action_bet{};
    double pot = 0, current_bet = 0, min_raise = 0;
    int to_act = -1, street = PREFLOP, button = 0, sb_player = 1, bb_player = 2;
    int last_full_raiser = -1;
    uint8_t pending_mask = 0;
    std::shared_ptr<const HistoryNode> history_tail;
    int history_size = 0;
    bool terminal = false;
    bool has_payoffs = false, has_payouts = false;
    std::array<double, PLAYERS> payoffs{}, payouts{};
    FixedVector<int, 3> winners;

    int players_remaining() const {
        return static_cast<int>(alive[0]) + static_cast<int>(alive[1]) + static_cast<int>(alive[2]);
    }
};

static std::vector<ActionRecord> history_vector(const State& s) {
    std::vector<ActionRecord> result(static_cast<std::size_t>(s.history_size));
    auto node = s.history_tail;
    for (int index = s.history_size - 1; index >= 0; --index) {
        if (!node) throw std::runtime_error("packed history chain is incomplete");
        result[static_cast<std::size_t>(index)] = node->event;
        node = node->previous;
    }
    if (node) throw std::runtime_error("packed history chain exceeds recorded size");
    return result;
}

static void append_history(State& s, ActionRecord event) {
    s.history_tail = std::make_shared<HistoryNode>(
        HistoryNode{std::move(event), s.history_tail}
    );
    ++s.history_size;
}

constexpr int CARD_FEATURES = 18;
constexpr int HISTORY_FEATURES = 4 + 3 + ACTIONS + 1;
constexpr int LEGACY_FIXED_FEATURES = 183;
constexpr int TOURNAMENT_FEATURES = 15;
constexpr int POKER_RELATIONAL_FEATURES = 66;

constexpr std::array<std::array<int, 5>, 10> STRAIGHT_WINDOWS{{
    {{12, 0, 1, 2, 3}},
    {{0, 1, 2, 3, 4}},
    {{1, 2, 3, 4, 5}},
    {{2, 3, 4, 5, 6}},
    {{3, 4, 5, 6, 7}},
    {{4, 5, 6, 7, 8}},
    {{5, 6, 7, 8, 9}},
    {{6, 7, 8, 9, 10}},
    {{7, 8, 9, 10, 11}},
    {{8, 9, 10, 11, 12}},
}};

static std::array<int, 10> straight_window_counts(
    const std::array<int, 13>& rank_present
) {
    std::array<int, 10> counts{};
    for (int window = 0; window < 10; ++window)
        for (int rank : STRAIGHT_WINDOWS[window])
            counts[window] += rank_present[rank];
    return counts;
}

static py::array_t<float> poker_relational_features_native(
    py::array_t<float, py::array::forcecast> cards_array,
    py::array_t<float, py::array::forcecast> street_array
) {
    if (cards_array.ndim() != 3 || cards_array.shape(1) != 7 || cards_array.shape(2) != 18)
        throw py::value_error("cards must have shape [batch, 7, 18]");
    if (street_array.ndim() != 2 || street_array.shape(0) != cards_array.shape(0)
        || street_array.shape(1) != 4)
        throw py::value_error("street_one_hot must have shape [batch, 4]");

    const py::ssize_t batch = cards_array.shape(0);
    auto cards = cards_array.unchecked<3>();
    auto streets = street_array.unchecked<2>();
    py::array_t<float> output({batch, static_cast<py::ssize_t>(POKER_RELATIONAL_FEATURES)});
    auto out = output.mutable_unchecked<2>();

    for (py::ssize_t row = 0; row < batch; ++row) {
        std::array<int, 13> rank_counts{}, board_rank_counts{}, rank_present{};
        std::array<int, 4> suit_counts{}, board_suit_counts{};
        std::array<std::array<int, 13>, 4> suited_rank_present{};
        std::array<int, 2> hole_rank{}, hole_suit{};
        int board_card_count = 0;

        for (int token = 0; token < 7; ++token) {
            if (!(cards(row, token, 17) > 0.0f)) continue;
            int rank = 0, suit = 0;
            float best_rank = cards(row, token, 0);
            float best_suit = cards(row, token, 13);
            for (int candidate = 1; candidate < 13; ++candidate) {
                if (cards(row, token, candidate) > best_rank) {
                    best_rank = cards(row, token, candidate); rank = candidate;
                }
            }
            for (int candidate = 1; candidate < 4; ++candidate) {
                if (cards(row, token, 13 + candidate) > best_suit) {
                    best_suit = cards(row, token, 13 + candidate); suit = candidate;
                }
            }
            ++rank_counts[rank]; ++suit_counts[suit];
            suited_rank_present[suit][rank] = 1;
            if (token < 2) {
                hole_rank[token] = rank; hole_suit[token] = suit;
            } else {
                ++board_rank_counts[rank]; ++board_suit_counts[suit];
                ++board_card_count;
            }
        }
        for (int rank = 0; rank < 13; ++rank) rank_present[rank] = rank_counts[rank] > 0;
        const auto straight_counts = straight_window_counts(rank_present);
        bool has_straight = false;
        for (int count : straight_counts) has_straight |= count >= 5;
        bool has_flush = false;
        for (int count : suit_counts) has_flush |= count >= 5;
        bool straight_flush = false;
        for (int suit = 0; suit < 4; ++suit) {
            const auto suited_counts = straight_window_counts(suited_rank_present[suit]);
            for (int count : suited_counts) straight_flush |= count >= 5;
        }

        int pair_count = 0, trip_count = 0;
        bool has_quads = false;
        for (int count : rank_counts) {
            pair_count += count >= 2; trip_count += count >= 3; has_quads |= count >= 4;
        }
        const bool has_full_house = trip_count >= 2 || (trip_count >= 1 && pair_count >= 2);
        const bool has_trips = trip_count >= 1;
        const bool has_two_pair = pair_count >= 2;
        const bool has_pair = pair_count >= 1;
        const std::array<bool, 9> category{{
            !(straight_flush || has_quads || has_full_house || has_flush || has_straight
              || has_trips || has_two_pair || has_pair),
            has_pair && !has_two_pair && !has_trips,
            has_two_pair && !has_trips,
            has_trips && !has_full_house && !has_straight && !has_flush,
            has_straight && !has_flush && !has_full_house && !has_quads,
            has_flush && !has_full_house && !has_quads && !straight_flush,
            has_full_house && !has_quads,
            has_quads && !straight_flush,
            straight_flush,
        }};

        int street_index = 0;
        for (int street = 1; street < 4; ++street)
            if (streets(row, street) > streets(row, street_index)) street_index = street;
        const bool can_draw = street_index < 3;
        bool has_straight_draw = false, open_ended = false, gutshot = false;
        for (int window = 0; window < 10; ++window) if (straight_counts[window] == 4) {
            has_straight_draw = true;
            int missing_position = -1;
            for (int position = 0; position < 5; ++position)
                if (!rank_present[STRAIGHT_WINDOWS[window][position]]) missing_position = position;
            open_ended |= missing_position == 0 || missing_position == 4;
            gutshot |= missing_position >= 1 && missing_position <= 3;
        }
        has_straight_draw = has_straight_draw && can_draw && !has_straight;
        open_ended = open_ended && can_draw && !has_straight;
        gutshot = gutshot && can_draw && !has_straight;
        bool flush_draw = false, backdoor_flush = false;
        for (int count : suit_counts) {
            flush_draw |= count == 4;
            backdoor_flush |= count == 3;
        }
        flush_draw = flush_draw && can_draw && !has_flush;
        backdoor_flush = street_index == 1 && backdoor_flush && !has_flush;

        const bool pocket_pair = hole_rank[0] == hole_rank[1];
        const bool hole_suited = hole_suit[0] == hole_suit[1];
        const int raw_gap = std::abs(hole_rank[0] - hole_rank[1]);
        const bool has_ace = hole_rank[0] == 12 || hole_rank[1] == 12;
        const int ace_low_gap = has_ace ? std::min(raw_gap, 13 - raw_gap) : raw_gap;
        const int gap_bucket = std::min(ace_low_gap, 4);
        const int hole_board_matches = (board_rank_counts[hole_rank[0]] > 0)
            + (board_rank_counts[hole_rank[1]] > 0);
        int board_max_rank = -1, board_pair_count = 0, board_trip_count = 0;
        for (int rank = 0; rank < 13; ++rank) {
            if (board_rank_counts[rank] > 0) board_max_rank = rank;
            board_pair_count += board_rank_counts[rank] >= 2;
            board_trip_count += board_rank_counts[rank] >= 3;
        }
        const int overcards = (hole_rank[0] > board_max_rank) + (hole_rank[1] > board_max_rank);
        const int max_board_suit = *std::max_element(board_suit_counts.begin(), board_suit_counts.end());

        int column = 0;
        for (int value : rank_counts) out(row, column++) = static_cast<float>(value) / 4.0f;
        for (int value : suit_counts) out(row, column++) = static_cast<float>(value) / 7.0f;
        for (int value : board_rank_counts) out(row, column++) = static_cast<float>(value) / 3.0f;
        for (int value : board_suit_counts) out(row, column++) = static_cast<float>(value) / 5.0f;
        for (bool value : category) out(row, column++) = value ? 1.0f : 0.0f;
        for (int value = 0; value < 5; ++value) out(row, column++) = value == gap_bucket ? 1.0f : 0.0f;
        const std::array<float, 18> scalar{{
            pocket_pair ? 1.0f : 0.0f,
            hole_suited ? 1.0f : 0.0f,
            static_cast<float>(hole_board_matches) / 2.0f,
            static_cast<float>(overcards) / 2.0f,
            has_straight_draw ? 1.0f : 0.0f,
            open_ended ? 1.0f : 0.0f,
            gutshot ? 1.0f : 0.0f,
            flush_draw ? 1.0f : 0.0f,
            backdoor_flush ? 1.0f : 0.0f,
            board_pair_count > 0 ? 1.0f : 0.0f,
            board_trip_count > 0 ? 1.0f : 0.0f,
            board_card_count >= 3 && max_board_suit == board_card_count ? 1.0f : 0.0f,
            board_card_count >= 3 && max_board_suit == 2 ? 1.0f : 0.0f,
            static_cast<float>(board_card_count) / 5.0f,
            static_cast<float>(pair_count) / 3.0f,
            static_cast<float>(trip_count) / 2.0f,
            has_flush ? 1.0f : 0.0f,
            has_straight ? 1.0f : 0.0f,
        }};
        for (float value : scalar) out(row, column++) = value;
        if (column != POKER_RELATIONAL_FEATURES)
            throw std::runtime_error("native relational feature width is incorrect");
    }
    return output;
}

static void append_card_features(std::vector<double>& out, int card) {
    const size_t start = out.size();
    out.resize(start + CARD_FEATURES, 0.0);
    if (card < 0) return;
    if (card >= 52) throw py::value_error("card index must be in [0, 51]");
    out[start + card % 13] = 1.0;
    out[start + 13 + card / 13] = 1.0;
    out[start + 17] = 1.0;
}

static py::array_t<float> encode_information_state_native(
    const State& s,
    int hero,
    const std::vector<int>& legal_actions,
    double stack_size,
    int max_history,
    bool include_tournament_features,
    py::object tournament_total_chips_obj
) {
    if (hero < 0 || hero >= PLAYERS) throw py::value_error("hero must be 0, 1, or 2");
    if (!(stack_size > 0)) throw py::value_error("stack_size must be positive");
    if (max_history <= 0) throw py::value_error("max_history must be positive");

    const int expected = LEGACY_FIXED_FEATURES + HISTORY_FEATURES * max_history
        + (include_tournament_features ? TOURNAMENT_FEATURES : 0);
    std::vector<double> values;
    values.reserve(expected);
    auto one_hot = [&](int selected, int width) {
        for (int i = 0; i < width; ++i) values.push_back(selected == i ? 1.0 : 0.0);
    };

    one_hot(s.street, 4);
    one_hot((s.button - hero + PLAYERS) % PLAYERS, PLAYERS);
    values.push_back(hero == s.button ? 1.0 : 0.0);
    values.push_back(hero == s.sb_player ? 1.0 : 0.0);
    values.push_back(hero == s.bb_player ? 1.0 : 0.0);

    for (int offset = 0; offset < PLAYERS; ++offset) {
        const int seat = (hero + offset) % PLAYERS;
        values.push_back(s.stacks[seat] / stack_size);
        values.push_back(s.total_contrib[seat] / stack_size);
        values.push_back(s.street_contrib[seat] / stack_size);
        values.push_back(s.folded[seat] ? 1.0 : 0.0);
        values.push_back(s.all_in[seat] ? 1.0 : 0.0);
        values.push_back((s.pending_mask & (1 << seat)) ? 1.0 : 0.0);
        values.push_back(s.raise_rights[seat] ? 1.0 : 0.0);
        values.push_back(s.has_last_action_bet[seat] ? s.last_action_bet[seat] / stack_size : 0.0);
        values.push_back(s.has_last_action_bet[seat] ? 1.0 : 0.0);
    }

    values.push_back(s.last_full_raiser < 0 ? 1.0 : 0.0);
    for (int i = 0; i < PLAYERS; ++i) {
        values.push_back(
            s.last_full_raiser >= 0
            && (s.last_full_raiser - hero + PLAYERS) % PLAYERS == i ? 1.0 : 0.0
        );
    }
    const double to_call = std::max(s.current_bet - s.street_contrib[hero], 0.0);
    int active_count = 0;
    for (bool folded : s.folded) active_count += !folded;
    values.push_back(s.pot / (3.0 * stack_size));
    values.push_back(s.current_bet / stack_size);
    values.push_back(s.min_raise / stack_size);
    values.push_back(to_call / stack_size);
    values.push_back(static_cast<double>(s.board.size()) / 5.0);
    values.push_back(static_cast<double>(active_count) / 3.0);
    values.push_back(static_cast<double>(std::popcount(s.pending_mask)) / 3.0);

    if (s.hole[hero].size() != 2) throw py::value_error("hero must have exactly two hole cards");
    std::array<int, 2> hero_cards{s.hole[hero][0], s.hole[hero][1]};
    std::sort(hero_cards.begin(), hero_cards.end());
    for (int card : hero_cards) append_card_features(values, card);
    std::vector<int> canonical_board = s.board.vector();
    if (canonical_board.size() >= 3) std::sort(canonical_board.begin(), canonical_board.begin() + 3);
    for (int i = 0; i < 5; ++i) {
        append_card_features(values, i < static_cast<int>(canonical_board.size()) ? canonical_board[i] : -1);
    }

    const int used_history = std::min(max_history, s.history_size);
    values.insert(values.end(), (max_history - used_history) * HISTORY_FEATURES, 0.0);
    std::vector<const ActionRecord*> recent_history(static_cast<std::size_t>(used_history));
    auto history_node = s.history_tail;
    for (int index = used_history - 1; index >= 0; --index) {
        if (!history_node) throw std::runtime_error("packed history chain is incomplete");
        recent_history[static_cast<std::size_t>(index)] = &history_node->event;
        history_node = history_node->previous;
    }
    for (const ActionRecord* event_ptr : recent_history) {
        const ActionRecord& event = *event_ptr;
        one_hot(event.street, 4);
        one_hot((event.player - hero + PLAYERS) % PLAYERS, PLAYERS);
        one_hot(event.action, ACTIONS);
        values.push_back(event.amount / stack_size);
    }
    std::array<bool, ACTIONS> legal{};
    for (int action : legal_actions) {
        if (action < 0 || action >= ACTIONS) throw py::value_error("legal action is outside valid range");
        legal[action] = true;
    }
    for (bool is_legal : legal) values.push_back(is_legal ? 1.0 : 0.0);

    if (include_tournament_features) {
        double inferred_total = 0.0;
        for (double chips : s.initial_stacks) inferred_total += std::max(0.0, chips);
        const double total_chips = tournament_total_chips_obj.is_none()
            ? inferred_total : tournament_total_chips_obj.cast<double>();
        if (!(total_chips > 0)) throw py::value_error("tournament_total_chips must be positive");
        if (inferred_total > total_chips + 1e-6)
            throw py::value_error("state starting stacks exceed tournament_total_chips");
        for (int offset = 0; offset < PLAYERS; ++offset)
            values.push_back(s.alive[(hero + offset) % PLAYERS] ? 1.0 : 0.0);
        for (int offset = 0; offset < PLAYERS; ++offset)
            values.push_back(std::max(0.0, s.initial_stacks[(hero + offset) % PLAYERS]) / total_chips);
        const double hero_behind = std::max(0.0, s.stacks[hero]);
        for (int offset = 0; offset < PLAYERS; ++offset) {
            const int seat = (hero + offset) % PLAYERS;
            double effective = 0.0;
            if (seat == hero) effective = s.alive[hero] ? hero_behind : 0.0;
            else if (s.alive[hero] && s.alive[seat])
                effective = std::min(hero_behind, std::max(0.0, s.stacks[seat]));
            values.push_back(effective / stack_size);
        }
        int players_remaining = 0, players_in_hand = 0;
        double shortest = std::numeric_limits<double>::infinity(), largest = 0.0;
        for (int seat = 0; seat < PLAYERS; ++seat) if (s.alive[seat]) {
            ++players_remaining;
            players_in_hand += !s.folded[seat];
            shortest = std::min(shortest, std::max(0.0, s.initial_stacks[seat]));
            largest = std::max(largest, std::max(0.0, s.initial_stacks[seat]));
        }
        if (!players_remaining) shortest = 0.0;
        values.push_back(static_cast<double>(players_remaining) / 3.0);
        values.push_back(static_cast<double>(players_in_hand) / 3.0);
        values.push_back(players_remaining == 2 ? 1.0 : 0.0);
        values.push_back(total_chips / (3.0 * stack_size));
        values.push_back(shortest / stack_size);
        values.push_back(largest / stack_size);
    }

    if (static_cast<int>(values.size()) != expected)
        throw std::runtime_error("native encoder produced an unexpected number of values");
    py::array_t<float> output(expected);
    auto destination = output.mutable_unchecked<1>();
    for (int i = 0; i < expected; ++i) destination(i) = static_cast<float>(values[i]);
    return output;
}

static void validate_cards(const std::vector<int>& cards, int expected) {
    if (static_cast<int>(cards.size()) != expected) throw py::value_error("wrong number of cards");
    std::array<bool, 52> seen{};
    for (int c : cards) {
        if (c < 0 || c >= 52) throw py::value_error("card indices must be in the range 0..51");
        if (seen[c]) throw py::value_error("duplicate cards are not valid");
        seen[c] = true;
    }
}

static int pack_score(int category, const std::vector<int>& kickers) {
    std::array<int, 6> fields{};
    fields[0] = category;
    for (size_t i = 0; i < kickers.size() && i < 5; ++i) fields[i + 1] = kickers[i];
    int score = 0;
    for (int value : fields) score = score * 15 + value;
    return score;
}

static int evaluate5_unchecked(const std::array<int, 5>& cards) {
    std::array<int, 15> counts{};
    std::array<int, 5> ranks{}, suits{};
    for (int i = 0; i < 5; ++i) {
        ranks[i] = cards[i] % 13 + 2;
        suits[i] = cards[i] / 13;
        ++counts[ranks[i]];
    }
    std::vector<int> unique;
    for (int rank = 14; rank >= 2; --rank) if (counts[rank]) unique.push_back(rank);
    int straight_high = 0;
    if (unique.size() == 5) {
        if (unique == std::vector<int>({14, 5, 4, 3, 2})) straight_high = 5;
        else if (unique.front() - unique.back() == 4) straight_high = unique.front();
    }
    bool flush = std::all_of(suits.begin() + 1, suits.end(), [&](int s){ return s == suits[0]; });
    if (flush && straight_high) return pack_score(8, {straight_high});
    std::vector<std::pair<int,int>> groups;
    for (int rank = 2; rank <= 14; ++rank) if (counts[rank]) groups.emplace_back(counts[rank], rank);
    std::sort(groups.rbegin(), groups.rend());
    if (groups[0].first == 4) {
        int quad = groups[0].second, kicker = 0;
        for (int rank : unique) if (rank != quad) kicker = std::max(kicker, rank);
        return pack_score(7, {quad, kicker});
    }
    if (groups[0].first == 3 && groups[1].first == 2) return pack_score(6, {groups[0].second, groups[1].second});
    if (flush) { auto sorted = std::vector<int>(ranks.begin(), ranks.end()); std::sort(sorted.rbegin(), sorted.rend()); return pack_score(5, sorted); }
    if (straight_high) return pack_score(4, {straight_high});
    if (groups[0].first == 3) {
        std::vector<int> k{groups[0].second};
        for (int rank : unique) if (rank != groups[0].second) k.push_back(rank);
        return pack_score(3, k);
    }
    std::vector<int> pairs;
    for (int rank = 14; rank >= 2; --rank) if (counts[rank] == 2) pairs.push_back(rank);
    if (pairs.size() == 2) {
        int kicker = 0; for (int rank = 14; rank >= 2; --rank) if (counts[rank] == 1) { kicker = rank; break; }
        return pack_score(2, {pairs[0], pairs[1], kicker});
    }
    if (pairs.size() == 1) {
        std::vector<int> k{pairs[0]}; for (int rank : unique) if (rank != pairs[0]) k.push_back(rank);
        return pack_score(1, k);
    }
    auto sorted = std::vector<int>(ranks.begin(), ranks.end()); std::sort(sorted.rbegin(), sorted.rend());
    return pack_score(0, sorted);
}

static int evaluate5(const std::vector<int>& cards) {
    validate_cards(cards, 5);
    std::array<int,5> a{}; std::copy(cards.begin(), cards.end(), a.begin());
    return evaluate5_unchecked(a);
}

static int evaluate7(const std::vector<int>& cards) {
    validate_cards(cards, 7);
    int best = -1;
    for (int a=0;a<3;++a) for(int b=a+1;b<4;++b) for(int c=b+1;c<5;++c)
    for(int d=c+1;d<6;++d) for(int e=d+1;e<7;++e)
        best = std::max(best, evaluate5_unchecked({cards[a],cards[b],cards[c],cards[d],cards[e]}));
    return best;
}

static std::vector<SidePot> side_pots(const std::array<double,3>& contrib, const std::array<bool,3>& folded) {
    std::vector<double> levels;
    for (double v : contrib) { if (v < -EPS) throw py::value_error("contributions cannot be negative"); if (v > EPS) levels.push_back(v); }
    std::sort(levels.begin(), levels.end()); levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    std::vector<SidePot> result; double previous=0;
    for(double level:levels){ SidePot p; p.cap=level; for(int i=0;i<3;++i) if(contrib[i]+EPS>=level){p.contributors.push_back(i);if(!folded[i])p.eligible.push_back(i);} p.amount=(level-previous)*p.contributors.size(); if(p.amount>EPS)result.push_back(p); previous=level; }
    return result;
}

struct Option { int action; int kind; double payment; double target; }; // kind 0 fold, 1 check, 2 commit

class Env {
public:
    double starting_stack, small_blind, big_blind, stack_size, sb, bb;
    py::object rng;
    int last_button = 2;

    Env(double stack=200, double small=1, double big=2, py::object seed=py::none())
        : starting_stack(stack), small_blind(small), big_blind(big), stack_size(stack), sb(small), bb(big) {
        if (stack <= 0) throw py::value_error("starting_stack must be positive");
        if (!(small > 0 && small < big)) throw py::value_error("blinds must satisfy 0 < small_blind < big_blind");
        rng = py::module_::import("random").attr("Random")(seed);
    }

    static uint8_t mask_from(const std::array<bool,3>& values) { uint8_t m=0; for(int i=0;i<3;++i)if(values[i])m|=uint8_t(1<<i); return m; }
    static int next_clockwise(int start, uint8_t mask) { for(int d=1;d<=3;++d){int p=(start+d)%3;if(mask&(1<<p))return p;} return -1; }
    static uint8_t can_act(const State& s) { uint8_t m=0;for(int i=0;i<3;++i)if(s.alive[i]&&!s.folded[i]&&!s.all_in[i]&&s.stacks[i]>EPS)m|=uint8_t(1<<i);return m; }

    State new_hand(py::object button_obj=py::none(), py::object stacks_obj=py::none(), py::object deck_obj=py::none()) {
        State s;
        if(stacks_obj.is_none()) s.initial_stacks.fill(starting_stack);
        else { auto v=stacks_obj.cast<std::vector<double>>();if(v.size()!=3)throw py::value_error("stacks must contain exactly three values");for(int i=0;i<3;++i){if(!std::isfinite(v[i])||v[i]<0)throw py::value_error("starting stacks must be finite and nonnegative");s.initial_stacks[i]=v[i]<=EPS?0:v[i];} }
        uint8_t live=0;for(int i=0;i<3;++i){s.alive[i]=s.initial_stacks[i]>EPS;s.eliminated[i]=!s.alive[i];s.folded[i]=s.eliminated[i];if(s.alive[i])live|=uint8_t(1<<i);}
        if(std::popcount(live)<2)throw py::value_error("at least two players must have a positive stack");
        int button=button_obj.is_none()?next_clockwise(last_button,live):button_obj.cast<int>();
        if(button<0||button>=3||!(live&(1<<button)))throw py::value_error("button must be assigned to a live player"); last_button=button;s.button=button;
        if(deck_obj.is_none()){std::vector<int> deck(52);for(int i=0;i<52;++i)deck[i]=i;py::list list=py::cast(deck);rng.attr("shuffle")(list);s.deck=list.cast<std::vector<int>>();}
        else{auto deck=deck_obj.cast<std::vector<int>>();validate_cards(deck,52);s.deck=deck;}
        if(std::popcount(live)==2){s.sb_player=button;s.bb_player=next_clockwise(button,live);}else{s.sb_player=next_clockwise(button,live);s.bb_player=next_clockwise(s.sb_player,live);}
        std::vector<int> order;int p=s.sb_player;for(int i=0;i<std::popcount(live);++i){order.push_back(p);p=next_clockwise(p,live);}for(int round=0;round<2;++round)for(int seat:order){s.hole[seat].push_back(s.deck.back());s.deck.pop_back();}
        s.stacks=s.initial_stacks;
        for(auto [seat,blind]:std::array<std::pair<int,double>,2>{{{s.sb_player,small_blind},{s.bb_player,big_blind}}}){double posted=std::min(s.stacks[seat],blind);s.stacks[seat]-=posted;s.total_contrib[seat]+=posted;s.street_contrib[seat]+=posted;}
        for(int i=0;i<3;++i){s.all_in[i]=s.alive[i]&&s.stacks[i]<=EPS;if(s.all_in[i])s.stacks[i]=0;s.raise_rights[i]=s.alive[i]&&!s.all_in[i];if(s.raise_rights[i])s.pending_mask|=uint8_t(1<<i);}
        s.pot=s.total_contrib[0]+s.total_contrib[1]+s.total_contrib[2];s.current_bet=big_blind;s.min_raise=big_blind;s.to_act=next_clockwise(s.bb_player,s.pending_mask);
        if(s.to_act<0)runout(s);return s;
    }

    double amount_to_call(const State& s, int player=-1) const {if(player<0)player=s.to_act;if(player<0)return 0;return std::max(0.0,s.current_bet-s.street_contrib[player]);}
    std::vector<Option> options(const State& s) const {
        if(s.terminal)return{};int player=s.to_act;if(player<0||!(s.pending_mask&(1<<player)))throw std::runtime_error("non-terminal state has no valid pending actor");
        double contribution=s.street_contrib[player],stack=s.stacks[player],to_call=amount_to_call(s,player);std::vector<Option> out;std::set<std::pair<int,int64_t>> seen;
        auto add=[&](int action,int kind,double payment,double target){auto key=std::make_pair(kind,(int64_t)std::llround(target*1e9));if(seen.insert(key).second)out.push_back({action,kind,payment,target});};
        if(to_call>EPS){add(FOLD,0,0,contribution);double pay=std::min(stack,to_call);add(CALL,2,pay,contribution+pay);}else add(CHECK,1,0,contribution);
        double max_target=contribution+stack;if(!(s.raise_rights[player]&&max_target>s.current_bet+EPS)){std::sort(out.begin(),out.end(),[](auto&a,auto&b){return a.action<b.action;});return out;}
        double minimum=s.current_bet+s.min_raise,base=s.current_bet>EPS?s.current_bet:big_blind;
        std::array<std::pair<int,double>,5> candidates{{{MIN_RAISE,minimum},{RAISE_2X,2*base},{RAISE_3X,3*base},{HALF_POT,contribution+to_call+std::max(.5*(s.pot+to_call),s.current_bet<=EPS?big_blind:0.)},{POT,contribution+to_call+std::max(s.pot+to_call,s.current_bet<=EPS?big_blind:0.)}}};
        for(auto[a,t]:candidates)if(t<=max_target+EPS&&t>=minimum-EPS){t=std::min(t,max_target);add(a,2,t-contribution,t);}add(ALL_IN,2,stack,max_target);std::sort(out.begin(),out.end(),[](auto&a,auto&b){return a.action<b.action;});return out;
    }
    std::vector<int> legal_actions(const State& s)const{std::vector<int> a;for(auto&o:options(s))a.push_back(o.action);return a;}
    std::vector<int> legal_mask(const State& s)const{std::vector<int> m(9);for(int a:legal_actions(s))m[a]=1;return m;}
    double action_target(const State&s,int action)const{for(auto&o:options(s))if(o.action==action)return o.target;throw py::value_error("illegal action");}

    State step(const State& old,int action){
        if(old.terminal)throw std::runtime_error("cannot act on a terminal state");Option selected{};bool found=false;auto opts=options(old);for(auto&o:opts)if(o.action==action){selected=o;found=true;}if(!found)throw py::value_error("illegal action");State s=old;int player=s.to_act;double before=s.current_bet;bool full=false;
        if(selected.kind==0){s.folded[player]=true;s.raise_rights[player]=false;s.has_last_action_bet[player]=true;s.last_action_bet[player]=s.current_bet;s.pending_mask&=uint8_t(~(1<<player));}
        else if(selected.kind==1){s.raise_rights[player]=false;s.has_last_action_bet[player]=true;s.last_action_bet[player]=s.current_bet;s.pending_mask&=uint8_t(~(1<<player));}
        else{double payment=std::min(selected.payment,s.stacks[player]);s.stacks[player]-=payment;if(s.stacks[player]<=EPS){s.stacks[player]=0;s.all_in[player]=true;}s.street_contrib[player]+=payment;s.total_contrib[player]+=payment;s.pot+=payment;double total=s.street_contrib[player];s.pending_mask&=uint8_t(~(1<<player));if(total>before+EPS){double increment=total-before;double old_min=s.min_raise;full=increment+EPS>=s.min_raise;s.current_bet=total;if(full){s.min_raise=increment;s.last_full_raiser=player;uint8_t active=can_act(s);for(int i=0;i<3;++i)if(i!=player&&(active&(1<<i)))s.raise_rights[i]=true;}else if(before<=EPS){uint8_t active=can_act(s);for(int i=0;i<3;++i)if(i!=player&&(active&(1<<i)))s.raise_rights[i]=true;}else{uint8_t active=can_act(s);for(int i=0;i<3;++i)if(i!=player&&(active&(1<<i))&&!s.raise_rights[i]&&s.has_last_action_bet[i]&&total-s.last_action_bet[i]+EPS>=old_min)s.raise_rights[i]=true;}s.pending_mask=0;uint8_t active=can_act(s);for(int i=0;i<3;++i)if(i!=player&&(active&(1<<i))&&s.street_contrib[i]+EPS<s.current_bet)s.pending_mask|=uint8_t(1<<i);}s.raise_rights[player]=false;s.has_last_action_bet[player]=true;s.last_action_bet[player]=s.current_bet;}
        append_history(s,{player,s.street,action,ACTION_NAMES[action],selected.payment,s.street_contrib[player],before,s.current_bet,s.pot,full});
        std::vector<int> remaining;for(int i=0;i<3;++i)if(s.alive[i]&&!s.folded[i])remaining.push_back(i);if(remaining.size()==1){award_uncontested(s,remaining[0]);return s;}
        s.pending_mask&=can_act(s);if(!s.pending_mask)close_round(s);else s.to_act=next_clockwise(player,s.pending_mask);assert_conservation(s);return s;
    }

    State resolve_showdown(const State& old){if(old.terminal)throw std::runtime_error("state is already terminal");if(old.board.size()!=5)throw py::value_error("showdown requires a five-card board");State s=old;showdown(s);return s;}
    double terminal_payoff(const State&s,int player)const{if(player<0||player>=3)throw py::value_error("player must be seat 0, 1, or 2");if(!s.terminal||!s.has_payoffs)throw std::runtime_error("payoff is available only at a terminal state");return s.payoffs[player];}

private:
    void burn(State&s){if(s.deck.empty())throw std::runtime_error("deck exhausted while burning");s.burned.push_back(s.deck.back());s.deck.pop_back();}
    void deal(State&s,int count){if((int)s.deck.size()<count)throw std::runtime_error("deck exhausted while dealing board");while(count--){s.board.push_back(s.deck.back());s.deck.pop_back();}}
    void advance(State&s){if(s.street==PREFLOP){burn(s);deal(s,3);s.street=FLOP;}else if(s.street==FLOP){burn(s);deal(s,1);s.street=TURN;}else if(s.street==TURN){burn(s);deal(s,1);s.street=RIVER;}else throw std::runtime_error("cannot advance beyond river");s.street_contrib.fill(0);s.current_bet=0;s.min_raise=big_blind;s.last_full_raiser=-1;s.has_last_action_bet.fill(false);s.last_action_bet.fill(0);uint8_t active=can_act(s);s.pending_mask=active;for(int i=0;i<3;++i)s.raise_rights[i]=active&(1<<i);s.to_act=next_clockwise(s.button,active);}
    void close_round(State&s){if(s.street==RIVER){showdown(s);return;}if(std::popcount(can_act(s))<2){runout(s);return;}advance(s);}
    void runout(State&s){while(s.board.size()<5){if(s.board.empty()){burn(s);deal(s,3);s.street=FLOP;}else if(s.board.size()==3){burn(s);deal(s,1);s.street=TURN;}else if(s.board.size()==4){burn(s);deal(s,1);s.street=RIVER;}else throw std::runtime_error("invalid board");}s.street=RIVER;showdown(s);}
    void finish(State&s,const std::array<double,3>& awards,const std::vector<int>& winners){s.pot=0;s.terminal=true;s.to_act=-1;s.pending_mask=0;s.payouts=awards;s.has_payouts=true;s.winners=winners;s.has_payoffs=true;for(int i=0;i<3;++i){s.payoffs[i]=s.stacks[i]-s.initial_stacks[i];s.alive[i]=s.stacks[i]>EPS;s.eliminated[i]=!s.alive[i];}assert_conservation(s);}
    void award_uncontested(State&s,int winner){std::array<double,3>a{};a[winner]=s.pot;s.stacks[winner]+=s.pot;finish(s,a,{winner});}
    void showdown(State&s){if(s.board.size()!=5)throw std::runtime_error("showdown requires five board cards");std::array<int,3>scores{};for(int i=0;i<3;++i)if(s.alive[i]&&!s.folded[i]){std::vector<int> cards=s.hole[i].vector();cards.insert(cards.end(),s.board.begin(),s.board.end());scores[i]=evaluate7(cards);}std::array<double,3>awards{};std::set<int> all;for(const auto&p:side_pots(s.total_contrib,s.folded)){if(p.eligible.empty())throw std::runtime_error("side pot has no eligible player");int best=-1;for(int i:p.eligible)best=std::max(best,scores[i]);std::vector<int>w;for(int i:p.eligible)if(scores[i]==best)w.push_back(i);double share=p.amount/w.size();for(int i:w){awards[i]+=share;all.insert(i);}}for(int i=0;i<3;++i)s.stacks[i]+=awards[i];finish(s,awards,std::vector<int>(all.begin(),all.end()));}
    static void assert_conservation(const State&s){double expected=s.initial_stacks[0]+s.initial_stacks[1]+s.initial_stacks[2],actual=s.stacks[0]+s.stacks[1]+s.stacks[2]+s.pot;if(std::abs(actual-expected)>1e-7)throw std::runtime_error("chip conservation failed");if(s.has_payoffs&&std::abs(s.payoffs[0]+s.payoffs[1]+s.payoffs[2])>1e-7)throw std::runtime_error("terminal payoffs are not zero-sum");}
};

template<typename T> static std::vector<T> array_vector(const std::array<T,3>& a){return {a[0],a[1],a[2]};}
template<typename T> static void assign3(std::array<T,3>&a,const std::vector<T>&v){if(v.size()!=3)throw py::value_error("field needs three values");std::copy(v.begin(),v.end(),a.begin());}

static py::dict state_to_dict(const State& s) {
    py::dict d;
    d["deck"] = s.deck.vector(); d["board"] = s.board.vector(); d["burned"] = s.burned.vector();
    std::vector<std::vector<int>> hole; for(const auto& cards:s.hole)hole.push_back(cards.vector()); d["hole"] = hole;
    d["stacks"] = array_vector(s.stacks); d["initial_stacks"] = array_vector(s.initial_stacks);
    d["total_contrib"] = array_vector(s.total_contrib); d["street_contrib"] = array_vector(s.street_contrib);
    d["folded"] = array_vector(s.folded); d["all_in"] = array_vector(s.all_in);
    d["raise_rights"] = array_vector(s.raise_rights); d["alive"] = array_vector(s.alive);
    d["eliminated"] = array_vector(s.eliminated); d["last_action_bet"] = array_vector(s.last_action_bet);
    d["has_last_action_bet"] = array_vector(s.has_last_action_bet);
    d["pot"] = s.pot; d["current_bet"] = s.current_bet; d["min_raise"] = s.min_raise;
    d["to_act"] = s.to_act; d["street"] = s.street; d["button"] = s.button;
    d["sb_player"] = s.sb_player; d["bb_player"] = s.bb_player;
    d["last_full_raiser"] = s.last_full_raiser; d["pending_mask"] = s.pending_mask;
    d["terminal"] = s.terminal; d["has_payoffs"] = s.has_payoffs; d["has_payouts"] = s.has_payouts;
    d["payoffs"] = array_vector(s.payoffs); d["payouts"] = array_vector(s.payouts); d["winners"] = s.winners.vector();
    py::list history;
    for (const auto& r : history_vector(s)) history.append(py::make_tuple(
        r.player, r.street, r.action, r.action_name, r.amount,
        r.contribution_after, r.current_bet_before, r.current_bet_after,
        r.pot_after, r.full_raise));
    d["history"] = history;
    return d;
}

static State state_from_dict(const py::dict& d) {
    State s;
    s.deck=d["deck"].cast<std::vector<int>>(); s.board=d["board"].cast<std::vector<int>>(); s.burned=d["burned"].cast<std::vector<int>>();
    auto hole=d["hole"].cast<std::vector<std::vector<int>>>(); if(hole.size()!=3)throw py::value_error("invalid native state");for(int i=0;i<3;++i)s.hole[i]=hole[i];
    assign3(s.stacks,d["stacks"].cast<std::vector<double>>()); assign3(s.initial_stacks,d["initial_stacks"].cast<std::vector<double>>());
    assign3(s.total_contrib,d["total_contrib"].cast<std::vector<double>>()); assign3(s.street_contrib,d["street_contrib"].cast<std::vector<double>>());
    assign3(s.folded,d["folded"].cast<std::vector<bool>>()); assign3(s.all_in,d["all_in"].cast<std::vector<bool>>());
    assign3(s.raise_rights,d["raise_rights"].cast<std::vector<bool>>()); assign3(s.alive,d["alive"].cast<std::vector<bool>>());
    assign3(s.eliminated,d["eliminated"].cast<std::vector<bool>>()); assign3(s.last_action_bet,d["last_action_bet"].cast<std::vector<double>>());
    assign3(s.has_last_action_bet,d["has_last_action_bet"].cast<std::vector<bool>>());
    s.pot=d["pot"].cast<double>();s.current_bet=d["current_bet"].cast<double>();s.min_raise=d["min_raise"].cast<double>();
    s.to_act=d["to_act"].cast<int>();s.street=d["street"].cast<int>();s.button=d["button"].cast<int>();s.sb_player=d["sb_player"].cast<int>();s.bb_player=d["bb_player"].cast<int>();s.last_full_raiser=d["last_full_raiser"].cast<int>();s.pending_mask=d["pending_mask"].cast<uint8_t>();
    s.terminal=d["terminal"].cast<bool>();s.has_payoffs=d["has_payoffs"].cast<bool>();s.has_payouts=d["has_payouts"].cast<bool>();
    assign3(s.payoffs,d["payoffs"].cast<std::vector<double>>());assign3(s.payouts,d["payouts"].cast<std::vector<double>>());s.winners=d["winners"].cast<std::vector<int>>();
    for(py::handle item:d["history"].cast<py::list>()){auto t=py::reinterpret_borrow<py::tuple>(item);append_history(s,{t[0].cast<int>(),t[1].cast<int>(),t[2].cast<int>(),t[3].cast<std::string>(),t[4].cast<double>(),t[5].cast<double>(),t[6].cast<double>(),t[7].cast<double>(),t[8].cast<double>(),t[9].cast<bool>()});}
    return s;
}

PYBIND11_MODULE(poker_native_engine, m) {
    m.doc()="Packed C++ three-player Hold'em engine";
    py::class_<ActionRecord>(m,"ActionRecord")
      .def_readonly("player",&ActionRecord::player).def_readonly("street",&ActionRecord::street).def_readonly("action",&ActionRecord::action).def_readonly("action_name",&ActionRecord::action_name).def_readonly("amount",&ActionRecord::amount).def_readonly("contribution_after",&ActionRecord::contribution_after).def_readonly("current_bet_before",&ActionRecord::current_bet_before).def_readonly("current_bet_after",&ActionRecord::current_bet_after).def_readonly("pot_after",&ActionRecord::pot_after).def_readonly("full_raise",&ActionRecord::full_raise);
    py::class_<SidePot>(m,"SidePot").def_readonly("amount",&SidePot::amount).def_readonly("cap",&SidePot::cap).def_property_readonly("contributors",[](const SidePot&p){return py::tuple(py::cast(p.contributors));}).def_property_readonly("eligible",[](const SidePot&p){return py::tuple(py::cast(p.eligible));});
    py::class_<State>(m,"ThreePlayerState")
      .def(py::init<>()).def_property("deck",[](const State&s){return s.deck.vector();},[](State&s,const std::vector<int>&v){s.deck=v;}).def_property("board",[](const State&s){return s.board.vector();},[](State&s,const std::vector<int>&v){s.board=v;}).def_property("burned",[](const State&s){return s.burned.vector();},[](State&s,const std::vector<int>&v){s.burned=v;}).def_property("hole",[](const State&s){std::vector<std::vector<int>>v;for(const auto&cards:s.hole)v.push_back(cards.vector());return v;},[](State&s,const std::vector<std::vector<int>>&v){if(v.size()!=3)throw py::value_error("hole needs three seats");for(int i=0;i<3;++i)s.hole[i]=v[i];})
      .def_property("stacks",[](const State&s){return array_vector(s.stacks);},[](State&s,const std::vector<double>&v){assign3(s.stacks,v);}).def_property("initial_stacks",[](const State&s){return array_vector(s.initial_stacks);},[](State&s,const std::vector<double>&v){assign3(s.initial_stacks,v);}).def_property("total_contrib",[](const State&s){return array_vector(s.total_contrib);},[](State&s,const std::vector<double>&v){assign3(s.total_contrib,v);}).def_property("street_contrib",[](const State&s){return array_vector(s.street_contrib);},[](State&s,const std::vector<double>&v){assign3(s.street_contrib,v);})
      .def_property("folded",[](const State&s){return array_vector(s.folded);},[](State&s,const std::vector<bool>&v){assign3(s.folded,v);}).def_property("all_in",[](const State&s){return array_vector(s.all_in);},[](State&s,const std::vector<bool>&v){assign3(s.all_in,v);}).def_property("raise_rights",[](const State&s){return array_vector(s.raise_rights);},[](State&s,const std::vector<bool>&v){assign3(s.raise_rights,v);}).def_property("alive",[](const State&s){return array_vector(s.alive);},[](State&s,const std::vector<bool>&v){assign3(s.alive,v);}).def_property_readonly("eliminated",[](const State&s){return array_vector(s.eliminated);})
      .def_property_readonly("pending_actors",[](const State&s){py::set x;for(int i=0;i<3;++i)if(s.pending_mask&(1<<i))x.add(i);return x;}).def_property_readonly("last_action_bet",[](const State&s){py::list x;for(int i=0;i<3;++i)x.append(s.has_last_action_bet[i]?py::cast(s.last_action_bet[i]):py::none());return x;}).def_property_readonly("last_full_raiser",[](const State&s)->py::object{return s.last_full_raiser<0?py::none():py::cast(s.last_full_raiser);})
      .def_readwrite("pot",&State::pot).def_readwrite("current_bet",&State::current_bet).def_readwrite("min_raise",&State::min_raise).def_property_readonly("to_act",[](const State&s)->py::object{return s.to_act<0?py::none():py::cast(s.to_act);}).def_readwrite("street",&State::street).def_readwrite("button",&State::button).def_readwrite("sb_player",&State::sb_player).def_readwrite("bb_player",&State::bb_player).def_property_readonly("history",&history_vector).def_readwrite("terminal",&State::terminal)
      .def_property_readonly("payoffs",[](const State&s)->py::object{return s.has_payoffs?py::cast(array_vector(s.payoffs)):py::none();}).def_property_readonly("payouts",[](const State&s)->py::object{return s.has_payouts?py::cast(array_vector(s.payouts)):py::none();}).def_property_readonly("winners",[](const State&s){return py::tuple(py::cast(s.winners.vector()));}).def_property_readonly("players_remaining",&State::players_remaining).def_property_readonly("contrib",[](const State&s){return array_vector(s.street_contrib);});
    py::class_<Env>(m,"ThreePlayerHoldemEnv")
      .def(py::init<double,double,double,py::object>(),py::arg("starting_stack")=200.,py::arg("small_blind")=1.,py::arg("big_blind")=2.,py::arg("seed")=py::none()).def_readonly("starting_stack",&Env::starting_stack).def_readonly("small_blind",&Env::small_blind).def_readonly("big_blind",&Env::big_blind).def_readonly("stack_size",&Env::stack_size).def_readonly("sb",&Env::sb).def_readonly("bb",&Env::bb).def_readwrite("rng",&Env::rng).def_property("_last_button",[](const Env&e){return e.last_button;},[](Env&e,int v){e.last_button=v;})
      .def("new_hand",&Env::new_hand,py::arg("button")=py::none(),py::kw_only(),py::arg("stacks")=py::none(),py::arg("deck")=py::none()).def_static("clone",[](const State&s){return s;}).def("amount_to_call",&Env::amount_to_call,py::arg("state"),py::arg("player")=-1).def("legal_actions",&Env::legal_actions).def("legal_action_mask",&Env::legal_mask).def("action_target",&Env::action_target).def("step",&Env::step).def("resolve_showdown",&Env::resolve_showdown).def("terminal_payoff",&Env::terminal_payoff);
    m.def("evaluate_5card",&evaluate5);m.def("evaluate_7card",&evaluate7);m.def("calculate_side_pots",[](const std::vector<double>&c,const std::vector<bool>&f){if(c.size()!=3||f.size()!=3)throw py::value_error("inputs need three values");std::array<double,3>ca{c[0],c[1],c[2]};std::array<bool,3>fa{f[0],f[1],f[2]};return side_pots(ca,fa);});
    m.def("state_to_dict", &state_to_dict); m.def("state_from_dict", &state_from_dict);
    m.def(
        "poker_relational_features", &poker_relational_features_native,
        py::arg("cards"), py::arg("street_one_hot")
    );
    m.def(
        "encode_information_state", &encode_information_state_native,
        py::arg("state"), py::arg("hero"), py::arg("legal_actions"),
        py::arg("stack_size"), py::arg("max_history") = 32,
        py::kw_only(), py::arg("include_tournament_features") = false,
        py::arg("tournament_total_chips") = py::none()
    );
}
