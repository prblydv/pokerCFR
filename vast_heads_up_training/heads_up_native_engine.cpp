#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

using Chip = std::int64_t;

constexpr int PLAYERS = 2;
constexpr int ACTIONS = 10;
constexpr int FOLD = 0;
constexpr int CHECK = 1;
constexpr int CALL = 2;
constexpr int MIN_RAISE = 3;
constexpr int THIRD_POT = 4;
constexpr int HALF_POT = 5;
constexpr int THREE_QUARTER_POT = 6;
constexpr int POT = 7;
constexpr int OVERBET = 8;
constexpr int ALL_IN = 9;
constexpr int PREFLOP = 0;
constexpr int FLOP = 1;
constexpr int TURN = 2;
constexpr int RIVER = 3;

const std::array<std::string, ACTIONS> ACTION_NAMES{{
    "fold",
    "check",
    "call",
    "min_raise",
    "third_pot",
    "half_pot",
    "three_quarter_pot",
    "pot",
    "overbet",
    "all_in",
}};

template <typename T, std::size_t Capacity>
class FixedVector {
public:
    FixedVector() = default;

    void assign(const std::vector<T>& values) {
        if (values.size() > Capacity) {
            throw py::value_error("packed field exceeds capacity");
        }
        size_ = values.size();
        std::copy(values.begin(), values.end(), data_.begin());
    }

    FixedVector& operator=(const std::vector<T>& values) {
        assign(values);
        return *this;
    }

    void push_back(const T& value) {
        if (size_ >= Capacity) {
            throw std::runtime_error("packed field capacity exceeded");
        }
        data_[size_++] = value;
    }

    void pop_back() {
        if (size_ == 0) {
            throw std::runtime_error("cannot pop an empty packed field");
        }
        --size_;
    }

    T& back() {
        if (size_ == 0) throw std::runtime_error("empty packed field");
        return data_[size_ - 1];
    }

    const T& back() const {
        if (size_ == 0) throw std::runtime_error("empty packed field");
        return data_[size_ - 1];
    }

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

struct ActionRecord {
    int player = 0;
    int street = PREFLOP;
    int action = -1;
    std::string kind;
    Chip amount = 0;
    Chip raise_to = 0;
    Chip contribution_after = 0;
    Chip current_bet_before = 0;
    Chip current_bet_after = 0;
    Chip pot_before = 0;
    Chip pot_after = 0;
    Chip to_call_before = 0;
    bool full_raise = false;
    bool all_in = false;
};

struct HistoryNode {
    ActionRecord event;
    std::shared_ptr<const HistoryNode> previous;
};

struct HeadsUpState {
    FixedVector<int, 52> deck;
    FixedVector<int, 5> board;
    FixedVector<int, 3> burned;
    std::array<FixedVector<int, 2>, PLAYERS> hole;
    std::array<Chip, PLAYERS> stacks{};
    std::array<Chip, PLAYERS> initial_stacks{};
    std::array<Chip, PLAYERS> total_contrib{};
    std::array<Chip, PLAYERS> street_contrib{};
    std::array<bool, PLAYERS> folded{};
    std::array<bool, PLAYERS> all_in{};
    std::array<bool, PLAYERS> raise_rights{};
    std::array<Chip, PLAYERS> last_action_bet{};
    std::array<bool, PLAYERS> has_last_action_bet{};
    std::array<Chip, PLAYERS> uncalled_returns{};
    Chip small_blind = 0;
    Chip big_blind = 0;
    Chip pot = 0;
    Chip current_bet = 0;
    Chip min_raise = 0;
    int to_act = -1;
    int street = PREFLOP;
    int button = 0;
    int sb_player = 0;
    int bb_player = 1;
    int last_full_raiser = -1;
    std::uint8_t pending_mask = 0;
    std::shared_ptr<const HistoryNode> history_tail;
    int history_size = 0;
    bool terminal = false;
    bool has_payoffs = false;
    bool has_payouts = false;
    std::array<Chip, PLAYERS> payoffs{};
    std::array<Chip, PLAYERS> payouts{};
    FixedVector<int, PLAYERS> winners;

    int players_remaining() const {
        return static_cast<int>(!folded[0]) + static_cast<int>(!folded[1]);
    }
};

static std::vector<ActionRecord> history_vector(const HeadsUpState& state) {
    std::vector<ActionRecord> result(static_cast<std::size_t>(state.history_size));
    auto node = state.history_tail;
    for (int index = state.history_size - 1; index >= 0; --index) {
        if (!node) {
            throw std::runtime_error("packed history chain is incomplete");
        }
        result[static_cast<std::size_t>(index)] = node->event;
        node = node->previous;
    }
    if (node) {
        throw std::runtime_error("packed history chain exceeds recorded size");
    }
    return result;
}

static void append_history(HeadsUpState& state, ActionRecord event) {
    state.history_tail = std::make_shared<HistoryNode>(
        HistoryNode{std::move(event), state.history_tail}
    );
    ++state.history_size;
}

static void validate_cards(const std::vector<int>& cards, int expected) {
    if (static_cast<int>(cards.size()) != expected) {
        throw py::value_error("wrong number of cards");
    }
    std::array<bool, 52> seen{};
    for (int card : cards) {
        if (card < 0 || card >= 52) {
            throw py::value_error("card indices must be in the range 0..51");
        }
        if (seen[card]) {
            throw py::value_error("duplicate cards are not valid");
        }
        seen[card] = true;
    }
}

static int pack_score(int category, const std::vector<int>& kickers) {
    std::array<int, 6> fields{};
    fields[0] = category;
    for (std::size_t index = 0; index < kickers.size() && index < 5; ++index) {
        fields[index + 1] = kickers[index];
    }
    int score = 0;
    for (int value : fields) score = score * 15 + value;
    return score;
}

static int evaluate_5card_unchecked(const std::array<int, 5>& cards) {
    std::array<int, 15> counts{};
    std::array<int, 5> ranks{};
    std::array<int, 5> suits{};
    for (int index = 0; index < 5; ++index) {
        ranks[index] = cards[index] % 13 + 2;
        suits[index] = cards[index] / 13;
        ++counts[ranks[index]];
    }

    std::vector<int> unique;
    for (int rank = 14; rank >= 2; --rank) {
        if (counts[rank]) unique.push_back(rank);
    }
    int straight_high = 0;
    if (unique.size() == 5) {
        if (unique == std::vector<int>{14, 5, 4, 3, 2}) straight_high = 5;
        else if (unique.front() - unique.back() == 4) straight_high = unique.front();
    }
    const bool flush = std::all_of(
        suits.begin() + 1,
        suits.end(),
        [&](int suit) { return suit == suits[0]; }
    );
    if (flush && straight_high) return pack_score(8, {straight_high});

    std::vector<std::pair<int, int>> groups;
    for (int rank = 2; rank <= 14; ++rank) {
        if (counts[rank]) groups.emplace_back(counts[rank], rank);
    }
    std::sort(groups.rbegin(), groups.rend());
    if (groups[0].first == 4) {
        int kicker = 0;
        for (int rank : unique) {
            if (rank != groups[0].second) kicker = std::max(kicker, rank);
        }
        return pack_score(7, {groups[0].second, kicker});
    }
    if (groups[0].first == 3 && groups[1].first == 2) {
        return pack_score(6, {groups[0].second, groups[1].second});
    }
    if (flush) {
        auto sorted = std::vector<int>(ranks.begin(), ranks.end());
        std::sort(sorted.rbegin(), sorted.rend());
        return pack_score(5, sorted);
    }
    if (straight_high) return pack_score(4, {straight_high});
    if (groups[0].first == 3) {
        std::vector<int> kickers{groups[0].second};
        for (int rank : unique) {
            if (rank != groups[0].second) kickers.push_back(rank);
        }
        return pack_score(3, kickers);
    }

    std::vector<int> pairs;
    for (int rank = 14; rank >= 2; --rank) {
        if (counts[rank] == 2) pairs.push_back(rank);
    }
    if (pairs.size() == 2) {
        int kicker = 0;
        for (int rank = 14; rank >= 2; --rank) {
            if (counts[rank] == 1) {
                kicker = rank;
                break;
            }
        }
        return pack_score(2, {pairs[0], pairs[1], kicker});
    }
    if (pairs.size() == 1) {
        std::vector<int> kickers{pairs[0]};
        for (int rank : unique) {
            if (rank != pairs[0]) kickers.push_back(rank);
        }
        return pack_score(1, kickers);
    }

    auto sorted = std::vector<int>(ranks.begin(), ranks.end());
    std::sort(sorted.rbegin(), sorted.rend());
    return pack_score(0, sorted);
}

static int evaluate_5card(const std::vector<int>& cards) {
    validate_cards(cards, 5);
    std::array<int, 5> packed{};
    std::copy(cards.begin(), cards.end(), packed.begin());
    return evaluate_5card_unchecked(packed);
}

static int evaluate_7card(const std::vector<int>& cards) {
    validate_cards(cards, 7);
    int best = -1;
    for (int a = 0; a < 3; ++a)
        for (int b = a + 1; b < 4; ++b)
            for (int c = b + 1; c < 5; ++c)
                for (int d = c + 1; d < 6; ++d)
                    for (int e = d + 1; e < 7; ++e)
                        best = std::max(
                            best,
                            evaluate_5card_unchecked(
                                {cards[a], cards[b], cards[c], cards[d], cards[e]}
                            )
                        );
    return best;
}

static py::dict bayesian_condition(
    const std::vector<double>& weights,
    const std::vector<double>& likelihoods,
    double likelihood_floor
) {
    if (weights.empty() || weights.size() != likelihoods.size()) {
        throw py::value_error(
            "weights and likelihoods must be non-empty and aligned"
        );
    }
    if (!std::isfinite(likelihood_floor) || likelihood_floor < 0.0) {
        throw py::value_error("likelihood_floor must be finite and nonnegative");
    }
    std::vector<double> posterior(weights.size(), 0.0);
    double total = 0.0;
    for (std::size_t index = 0; index < weights.size(); ++index) {
        if (
            !std::isfinite(weights[index]) || weights[index] < 0.0
            || !std::isfinite(likelihoods[index]) || likelihoods[index] < 0.0
        ) {
            throw py::value_error(
                "weights and likelihoods must be finite and nonnegative"
            );
        }
        posterior[index] = weights[index]
            * std::max(likelihood_floor, likelihoods[index]);
        total += posterior[index];
    }
    if (!std::isfinite(total) || total <= 0.0) {
        throw py::value_error("Bayesian posterior has zero or invalid mass");
    }
    double square_sum = 0.0;
    for (double& value : posterior) {
        value /= total;
        square_sum += value * value;
    }
    py::dict result;
    result["weights"] = posterior;
    result["effective_sample_size"] = square_sum > 0.0
        ? 1.0 / square_sum
        : 0.0;
    return result;
}

static std::vector<double> regret_match_root(
    const std::vector<double>& regrets,
    const std::vector<bool>& allowed,
    const std::vector<double>& value_scores
) {
    if (
        regrets.empty() || regrets.size() != allowed.size()
        || regrets.size() != value_scores.size()
    ) {
        throw py::value_error(
            "regrets, allowed mask, and value scores must be aligned"
        );
    }
    std::vector<double> strategy(regrets.size(), 0.0);
    double positive_total = 0.0;
    for (std::size_t index = 0; index < regrets.size(); ++index) {
        if (
            !std::isfinite(regrets[index])
            || (!std::isfinite(value_scores[index])
                && value_scores[index] != -std::numeric_limits<double>::infinity())
        ) {
            throw py::value_error("root regrets and values must be valid");
        }
        if (allowed[index]) {
            strategy[index] = std::max(0.0, regrets[index]);
            positive_total += strategy[index];
        }
    }
    if (positive_total > 1e-12) {
        for (double& value : strategy) value /= positive_total;
        return strategy;
    }

    std::size_t best = regrets.size();
    double best_value = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < value_scores.size(); ++index) {
        if (allowed[index] && value_scores[index] > best_value) {
            best = index;
            best_value = value_scores[index];
        }
    }
    if (best == regrets.size()) {
        throw py::value_error("root strategy has no allowed action");
    }
    strategy[best] = 1.0;
    return strategy;
}

static py::dict hierarchical_regret_match_root(
    const std::vector<double>& regrets,
    const std::vector<bool>& allowed,
    const std::vector<double>& value_scores,
    const std::vector<int>& families
) {
    if (
        regrets.empty() || regrets.size() != allowed.size()
        || regrets.size() != value_scores.size()
        || regrets.size() != families.size()
    ) {
        throw py::value_error(
            "regrets, allowed mask, values, and families must be aligned"
        );
    }
    constexpr int FAMILY_COUNT = 3;
    std::array<double, FAMILY_COUNT> family_regrets{};
    std::array<double, FAMILY_COUNT> family_values{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()
    };
    std::array<bool, FAMILY_COUNT> family_allowed{};
    for (std::size_t index = 0; index < regrets.size(); ++index) {
        const int family = families[index];
        if (family < 0 || family >= FAMILY_COUNT) {
            throw py::value_error("family ids must be fold=0, passive=1, raise=2");
        }
        if (!allowed[index]) continue;
        family_allowed[family] = true;
        family_regrets[family] += std::max(0.0, regrets[index]);
        family_values[family] = std::max(
            family_values[family], value_scores[index]
        );
    }
    const std::vector<double> family_regret_vector(
        family_regrets.begin(), family_regrets.end()
    );
    const std::vector<bool> family_allowed_vector(
        family_allowed.begin(), family_allowed.end()
    );
    const std::vector<double> family_value_vector(
        family_values.begin(), family_values.end()
    );
    const auto family_strategy = regret_match_root(
        family_regret_vector, family_allowed_vector, family_value_vector
    );

    std::vector<double> action_strategy(regrets.size(), 0.0);
    for (int family = 0; family < FAMILY_COUNT; ++family) {
        if (!family_allowed[family] || family_strategy[family] <= 0.0) continue;
        double total = 0.0;
        for (std::size_t index = 0; index < regrets.size(); ++index) {
            if (allowed[index] && families[index] == family) {
                action_strategy[index] = std::max(0.0, regrets[index]);
                total += action_strategy[index];
            }
        }
        if (total <= 1e-12) {
            std::size_t best = regrets.size();
            double best_value = -std::numeric_limits<double>::infinity();
            for (std::size_t index = 0; index < regrets.size(); ++index) {
                if (
                    allowed[index] && families[index] == family
                    && value_scores[index] > best_value
                ) {
                    best = index;
                    best_value = value_scores[index];
                }
            }
            if (best != regrets.size()) action_strategy[best] = 1.0;
            total = 1.0;
        }
        for (std::size_t index = 0; index < regrets.size(); ++index) {
            if (allowed[index] && families[index] == family) {
                action_strategy[index] =
                    family_strategy[family] * action_strategy[index] / total;
            }
        }
    }
    py::dict result;
    result["family_strategy"] = family_strategy;
    result["action_strategy"] = action_strategy;
    return result;
}

static py::dict estimate_terminal_call_scenarios(
    const std::vector<int>& hero_hole,
    const std::vector<int>& board,
    const std::vector<std::vector<int>>& opponent_holes,
    const std::vector<double>& weights,
    double fold_payoff,
    double win_payoff,
    double tie_payoff,
    double loss_payoff,
    std::uint64_t nominal_samples,
    std::uint64_t seed
) {
    validate_cards(hero_hole, 2);
    if (board.size() < 3 || board.size() > 5) {
        throw py::value_error(
            "terminal call evaluation requires a flop, turn, or river"
        );
    }
    std::vector<int> public_cards = hero_hole;
    public_cards.insert(public_cards.end(), board.begin(), board.end());
    validate_cards(public_cards, static_cast<int>(public_cards.size()));
    if (
        opponent_holes.empty() || opponent_holes.size() != weights.size()
        || nominal_samples == 0
    ) {
        throw py::value_error(
            "opponent range, weights, and samples must be valid"
        );
    }
    const double weight_total = std::accumulate(
        weights.begin(), weights.end(), 0.0
    );
    if (!std::isfinite(weight_total) || weight_total <= 0.0) {
        throw py::value_error("opponent weights must have positive mass");
    }
    for (const auto& opponent : opponent_holes) {
        if (opponent.size() != 2) {
            throw py::value_error("every opponent hand needs two cards");
        }
        std::vector<int> cards = public_cards;
        cards.insert(cards.end(), opponent.begin(), opponent.end());
        validate_cards(cards, static_cast<int>(cards.size()));
    }

    std::vector<double> win_probabilities(opponent_holes.size(), 0.0);
    std::vector<double> tie_probabilities(opponent_holes.size(), 0.0);
    {
        py::gil_scoped_release release;
        const int missing = 5 - static_cast<int>(board.size());
        for (std::size_t hand_index = 0;
             hand_index < opponent_holes.size();
             ++hand_index) {
            const auto& opponent = opponent_holes[hand_index];
            std::vector<int> cards = public_cards;
            cards.insert(cards.end(), opponent.begin(), opponent.end());
            std::array<bool, 52> blocked{};
            for (int card : cards) blocked[card] = true;
            std::vector<int> pool;
            for (int card = 0; card < 52; ++card) {
                if (!blocked[card]) pool.push_back(card);
            }
            std::uint64_t wins = 0;
            std::uint64_t ties = 0;
            std::uint64_t runouts = 0;
            auto score_runout = [&](const std::vector<int>& extra) {
                std::vector<int> hero_cards = hero_hole;
                std::vector<int> opponent_cards = opponent;
                hero_cards.insert(hero_cards.end(), board.begin(), board.end());
                opponent_cards.insert(
                    opponent_cards.end(), board.begin(), board.end()
                );
                hero_cards.insert(hero_cards.end(), extra.begin(), extra.end());
                opponent_cards.insert(
                    opponent_cards.end(), extra.begin(), extra.end()
                );
                const int hero_score = evaluate_7card(hero_cards);
                const int opponent_score = evaluate_7card(opponent_cards);
                wins += hero_score > opponent_score;
                ties += hero_score == opponent_score;
                ++runouts;
            };
            if (missing == 0) {
                score_runout({});
            } else if (missing == 1) {
                for (int card : pool) score_runout({card});
            } else {
                constexpr int FLOP_RUNOUTS_PER_HAND = 24;
                std::mt19937_64 equity_rng(
                    seed
                    ^ (
                        static_cast<std::uint64_t>(opponent[0] + 1) << 16
                    )
                    ^ static_cast<std::uint64_t>(opponent[1] + 1)
                );
                for (int sample = 0;
                     sample < FLOP_RUNOUTS_PER_HAND;
                     ++sample) {
                    std::uniform_int_distribution<std::size_t> first_card(
                        0, pool.size() - 1
                    );
                    const std::size_t first = first_card(equity_rng);
                    std::uniform_int_distribution<std::size_t> second_card(
                        0, pool.size() - 2
                    );
                    std::size_t second = second_card(equity_rng);
                    if (second >= first) ++second;
                    score_runout({pool[first], pool[second]});
                }
            }
            win_probabilities[hand_index] =
                static_cast<double>(wins) / static_cast<double>(runouts);
            tie_probabilities[hand_index] =
                static_cast<double>(ties) / static_cast<double>(runouts);
        }
    }

    std::vector<double> posterior(weights.size(), 0.0);
    std::vector<double> tempered(weights.size(), 0.0);
    std::vector<double> contaminated(weights.size(), 0.0);
    std::vector<double> value_heavy(weights.size(), 0.0);
    const double uniform = 1.0 / static_cast<double>(weights.size());
    for (std::size_t index = 0; index < weights.size(); ++index) {
        posterior[index] = weights[index] / weight_total;
        tempered[index] = std::sqrt(std::max(1e-18, posterior[index]));
        contaminated[index] = 0.75 * posterior[index] + 0.25 * uniform;
        const double hero_equity =
            win_probabilities[index] + 0.5 * tie_probabilities[index];
        value_heavy[index] = posterior[index] * std::exp(
            6.0 * (0.5 - hero_equity)
        );
    }
    auto normalize = [](std::vector<double>& values) {
        const double total = std::accumulate(
            values.begin(), values.end(), 0.0
        );
        if (!std::isfinite(total) || total <= 0.0) {
            throw std::runtime_error("range scenario has zero mass");
        }
        for (double& value : values) value /= total;
    };
    normalize(tempered);
    normalize(contaminated);
    normalize(value_heavy);

    const std::array<std::string, 4> names = {
        "posterior", "tempered", "contaminated", "value_heavy"
    };
    const std::array<const std::vector<double>*, 4> scenarios = {
        &posterior, &tempered, &contaminated, &value_heavy
    };
    py::list rows;
    double worst_mean = std::numeric_limits<double>::infinity();
    std::string worst_name;
    for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
        double mean = 0.0;
        double second_moment = 0.0;
        double equity = 0.0;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const double win = win_probabilities[index];
            const double tie = tie_probabilities[index];
            const double loss = std::max(0.0, 1.0 - win - tie);
            const double hand_mean =
                win * win_payoff + tie * tie_payoff + loss * loss_payoff;
            const double hand_second =
                win * win_payoff * win_payoff
                + tie * tie_payoff * tie_payoff
                + loss * loss_payoff * loss_payoff;
            const double weight = (*scenarios[scenario])[index];
            mean += weight * hand_mean;
            second_moment += weight * hand_second;
            equity += weight * (win + 0.5 * tie);
        }
        const double variance = std::max(
            0.0, second_moment - mean * mean
        );
        const double standard_error = std::sqrt(
            variance / static_cast<double>(nominal_samples)
        );
        py::dict row;
        row["name"] = names[scenario];
        row["mean"] = mean;
        row["standard_error"] = standard_error;
        row["ci95_low"] = mean - 1.96 * standard_error;
        row["ci95_high"] = mean + 1.96 * standard_error;
        row["equity"] = equity;
        rows.append(row);
        if (mean < worst_mean) {
            worst_mean = mean;
            worst_name = names[scenario];
        }
    }
    py::dict result;
    result["fold_payoff"] = fold_payoff;
    result["worst_mean"] = worst_mean;
    result["worst_name"] = worst_name;
    result["scenarios"] = rows;
    result["samples"] = nominal_samples;
    return result;
}

static py::dict estimate_all_in_ev(
    const std::vector<int>& hero_hole,
    const std::vector<int>& board,
    const std::vector<std::vector<int>>& opponent_holes,
    const std::vector<double>& weights,
    const std::vector<double>& call_probabilities,
    double fold_payoff,
    double win_payoff,
    double tie_payoff,
    double loss_payoff,
    std::uint64_t samples,
    std::uint64_t seed,
    bool robust_best_response
) {
    validate_cards(hero_hole, 2);
    if (board.size() > 5) {
        throw py::value_error("board cannot contain more than five cards");
    }
    std::vector<int> public_cards = hero_hole;
    public_cards.insert(public_cards.end(), board.begin(), board.end());
    validate_cards(public_cards, static_cast<int>(public_cards.size()));
    if (opponent_holes.empty()) {
        throw py::value_error("opponent range cannot be empty");
    }
    if (
        opponent_holes.size() != weights.size()
        || weights.size() != call_probabilities.size()
    ) {
        throw py::value_error(
            "opponent holes, weights, and call probabilities must align"
        );
    }
    if (samples == 0) {
        throw py::value_error("samples must be positive");
    }
    for (std::size_t index = 0; index < opponent_holes.size(); ++index) {
        if (opponent_holes[index].size() != 2) {
            throw py::value_error("every opponent hand needs two cards");
        }
        std::vector<int> row = public_cards;
        row.insert(
            row.end(),
            opponent_holes[index].begin(),
            opponent_holes[index].end()
        );
        validate_cards(row, static_cast<int>(row.size()));
        if (
            !std::isfinite(weights[index]) || weights[index] < 0.0
            || !std::isfinite(call_probabilities[index])
            || call_probabilities[index] < 0.0
            || call_probabilities[index] > 1.0
        ) {
            throw py::value_error(
                "range weights and call probabilities are invalid"
            );
        }
    }
    if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0) {
        throw py::value_error("range weights must have positive mass");
    }

    std::vector<double> effective_call_probabilities = call_probabilities;
    std::vector<double> robust_win_probabilities(
        opponent_holes.size(), 0.0
    );
    std::vector<double> robust_tie_probabilities(
        opponent_holes.size(), 0.0
    );
    std::uint64_t robust_call_hands = 0;
    const bool robust_enabled = robust_best_response && board.size() >= 3;
    if (robust_enabled) {
        py::gil_scoped_release release;
        const int missing = 5 - static_cast<int>(board.size());
        for (std::size_t hand_index = 0;
             hand_index < opponent_holes.size();
             ++hand_index) {
            const auto& opponent = opponent_holes[hand_index];
            std::array<bool, 52> blocked{};
            for (int card : hero_hole) blocked[card] = true;
            for (int card : board) blocked[card] = true;
            for (int card : opponent) blocked[card] = true;
            std::vector<int> pool;
            for (int card = 0; card < 52; ++card) {
                if (!blocked[card]) pool.push_back(card);
            }
            std::uint64_t wins = 0;
            std::uint64_t ties = 0;
            std::uint64_t runouts = 0;
            auto score_runout = [&](const std::vector<int>& extra) {
                std::vector<int> hero_cards = hero_hole;
                std::vector<int> opponent_cards = opponent;
                hero_cards.insert(hero_cards.end(), board.begin(), board.end());
                opponent_cards.insert(
                    opponent_cards.end(), board.begin(), board.end()
                );
                hero_cards.insert(hero_cards.end(), extra.begin(), extra.end());
                opponent_cards.insert(
                    opponent_cards.end(), extra.begin(), extra.end()
                );
                const int hero_score = evaluate_7card(hero_cards);
                const int opponent_score = evaluate_7card(opponent_cards);
                wins += hero_score > opponent_score;
                ties += hero_score == opponent_score;
                ++runouts;
            };
            if (missing == 0) {
                score_runout({});
            } else if (missing == 1) {
                for (int card : pool) score_runout({card});
            } else if (missing == 2) {
                // A full flop enumeration is about one million range/runout
                // pairs and violates the live-search deadline.  Use a
                // deterministic per-hand sample instead, so every range hand
                // receives equal equity work without card-order bias.
                constexpr int ROBUST_FLOP_RUNOUTS_PER_HAND = 24;
                std::mt19937_64 equity_rng(
                    seed
                    ^ (
                        static_cast<std::uint64_t>(opponent[0] + 1) << 16
                    )
                    ^ static_cast<std::uint64_t>(opponent[1] + 1)
                );
                for (int sample = 0;
                     sample < ROBUST_FLOP_RUNOUTS_PER_HAND;
                     ++sample) {
                    std::uniform_int_distribution<std::size_t> choose_first(
                        0, pool.size() - 1
                    );
                    const std::size_t first = choose_first(equity_rng);
                    std::uniform_int_distribution<std::size_t> choose_second(
                        0, pool.size() - 2
                    );
                    std::size_t second = choose_second(equity_rng);
                    if (second >= first) ++second;
                    score_runout({pool[first], pool[second]});
                }
            }
            if (runouts > 0) {
                const double win_probability =
                    static_cast<double>(wins) / static_cast<double>(runouts);
                const double tie_probability =
                    static_cast<double>(ties) / static_cast<double>(runouts);
                robust_win_probabilities[hand_index] = win_probability;
                robust_tie_probabilities[hand_index] = tie_probability;
                const double hero_showdown_ev =
                    win_probability * win_payoff
                    + tie_probability * tie_payoff
                    + (1.0 - win_probability - tie_probability) * loss_payoff;
                if (hero_showdown_ev <= fold_payoff + 1e-12) {
                    effective_call_probabilities[hand_index] = 1.0;
                    ++robust_call_hands;
                }
            }
        }
    }

    if (robust_enabled) {
        const double weight_total = std::accumulate(
            weights.begin(), weights.end(), 0.0
        );
        double mean = 0.0;
        double second_moment = 0.0;
        double call_mass = 0.0;
        double called_win_mass = 0.0;
        double called_tie_mass = 0.0;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const double normalized_weight = weights[index] / weight_total;
            const double call = effective_call_probabilities[index];
            const double win = robust_win_probabilities[index];
            const double tie = robust_tie_probabilities[index];
            const double loss = std::max(0.0, 1.0 - win - tie);
            const double showdown_mean =
                win * win_payoff + tie * tie_payoff + loss * loss_payoff;
            const double showdown_second =
                win * win_payoff * win_payoff
                + tie * tie_payoff * tie_payoff
                + loss * loss_payoff * loss_payoff;
            mean += normalized_weight * (
                (1.0 - call) * fold_payoff + call * showdown_mean
            );
            second_moment += normalized_weight * (
                (1.0 - call) * fold_payoff * fold_payoff
                + call * showdown_second
            );
            call_mass += normalized_weight * call;
            called_win_mass += normalized_weight * call * win;
            called_tie_mass += normalized_weight * call * tie;
        }
        const double variance = std::max(
            0.0, second_moment - mean * mean
        );
        const double standard_error = std::sqrt(
            variance / static_cast<double>(samples)
        );
        py::dict result;
        result["mean"] = mean;
        result["standard_error"] = standard_error;
        result["ci95_low"] = mean - 1.96 * standard_error;
        result["ci95_high"] = mean + 1.96 * standard_error;
        result["samples"] = samples;
        result["calls"] = static_cast<std::uint64_t>(
            std::llround(call_mass * static_cast<double>(samples))
        );
        result["call_rate"] = call_mass;
        result["called_equity"] = call_mass > 0.0
            ? (called_win_mass + 0.5 * called_tie_mass) / call_mass
            : 0.0;
        result["robust_best_response"] = true;
        result["robust_call_hands"] = robust_call_hands;
        return result;
    }

    double mean = 0.0;
    double m2 = 0.0;
    std::uint64_t calls = 0;
    std::uint64_t wins = 0;
    std::uint64_t ties = 0;
    {
        py::gil_scoped_release release;
        std::mt19937_64 rng(seed);
        std::discrete_distribution<std::size_t> choose_hand(
            weights.begin(),
            weights.end()
        );
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const int missing = 5 - static_cast<int>(board.size());
        for (std::uint64_t sample = 0; sample < samples; ++sample) {
            const std::size_t hand_index = choose_hand(rng);
            double value = fold_payoff;
            if (uniform(rng) < effective_call_probabilities[hand_index]) {
                ++calls;
                const auto& opponent = opponent_holes[hand_index];
                std::array<bool, 52> blocked{};
                for (int card : hero_hole) blocked[card] = true;
                for (int card : board) blocked[card] = true;
                for (int card : opponent) blocked[card] = true;
                std::vector<int> pool;
                pool.reserve(45);
                for (int card = 0; card < 52; ++card) {
                    if (!blocked[card]) pool.push_back(card);
                }
                for (int offset = 0; offset < missing; ++offset) {
                    std::uniform_int_distribution<std::size_t> choose_card(
                        static_cast<std::size_t>(offset),
                        pool.size() - 1
                    );
                    const auto selected = choose_card(rng);
                    std::swap(
                        pool[static_cast<std::size_t>(offset)],
                        pool[selected]
                    );
                }
                std::vector<int> hero_cards = hero_hole;
                std::vector<int> opponent_cards = opponent;
                hero_cards.insert(hero_cards.end(), board.begin(), board.end());
                opponent_cards.insert(
                    opponent_cards.end(), board.begin(), board.end()
                );
                hero_cards.insert(
                    hero_cards.end(), pool.begin(), pool.begin() + missing
                );
                opponent_cards.insert(
                    opponent_cards.end(), pool.begin(), pool.begin() + missing
                );
                const int hero_score = evaluate_7card(hero_cards);
                const int opponent_score = evaluate_7card(opponent_cards);
                if (hero_score > opponent_score) {
                    value = win_payoff;
                    ++wins;
                } else if (hero_score < opponent_score) {
                    value = loss_payoff;
                } else {
                    value = tie_payoff;
                    ++ties;
                }
            }
            const double delta = value - mean;
            mean += delta / static_cast<double>(sample + 1);
            m2 += delta * (value - mean);
        }
    }
    const double variance = samples > 1
        ? m2 / static_cast<double>(samples - 1)
        : 0.0;
    const double standard_error = std::sqrt(
        variance / static_cast<double>(samples)
    );
    const double called_equity = calls > 0
        ? (
            static_cast<double>(wins)
            + 0.5 * static_cast<double>(ties)
        ) / static_cast<double>(calls)
        : 0.0;
    py::dict result;
    result["mean"] = mean;
    result["standard_error"] = standard_error;
    result["ci95_low"] = mean - 1.96 * standard_error;
    result["ci95_high"] = mean + 1.96 * standard_error;
    result["samples"] = samples;
    result["calls"] = calls;
    result["call_rate"] = static_cast<double>(calls)
        / static_cast<double>(samples);
    result["called_equity"] = called_equity;
    result["robust_best_response"] = robust_enabled;
    result["robust_call_hands"] = robust_call_hands;
    return result;
}

enum class OptionKind {
    Fold,
    Check,
    Commit,
};

struct ActionOption {
    int action = -1;
    OptionKind kind = OptionKind::Check;
    std::string semantic;
    Chip payment = 0;
    Chip target = 0;
};

static Chip rounded_fraction(Chip value, Chip numerator, Chip denominator) {
    if (value < 0 || numerator < 0 || denominator <= 0) {
        throw std::runtime_error("invalid nonnegative chip fraction");
    }
    if (
        numerator != 0
        && value > (std::numeric_limits<Chip>::max() - denominator / 2)
            / numerator
    ) {
        throw std::overflow_error("chip amount is too large to size a raise");
    }
    const Chip product = value * numerator;
    return (product + denominator / 2) / denominator;
}

class HeadsUpHoldemEngine {
public:
    Chip starting_stack;
    Chip small_blind;
    Chip big_blind;
    Chip stack_size;
    Chip sb;
    Chip bb;
    py::object rng;
    int last_button = 1;

    HeadsUpHoldemEngine(
        Chip stack = 200,
        Chip small = 1,
        Chip big = 2,
        py::object seed = py::none()
    )
        : starting_stack(stack),
          small_blind(small),
          big_blind(big),
          stack_size(stack),
          sb(small),
          bb(big) {
        if (stack <= 0) throw py::value_error("starting_stack must be positive");
        if (!(small > 0 && small < big)) {
            throw py::value_error(
                "blinds must satisfy 0 < small_blind < big_blind"
            );
        }
        rng = py::module_::import("random").attr("Random")(seed);
    }

    HeadsUpState new_hand(
        py::object button_object = py::none(),
        py::object stacks_object = py::none(),
        py::object deck_object = py::none()
    ) {
        HeadsUpState state;
        if (stacks_object.is_none()) {
            state.initial_stacks.fill(starting_stack);
        } else {
            const auto values = stacks_object.cast<std::vector<Chip>>();
            if (values.size() != PLAYERS) {
                throw py::value_error("stacks must contain exactly two values");
            }
            for (int player = 0; player < PLAYERS; ++player) {
                if (values[player] <= 0) {
                    throw py::value_error("both starting stacks must be positive");
                }
                state.initial_stacks[player] = values[player];
            }
        }

        int button;
        if (button_object.is_none()) {
            button = 1 - last_button;
        } else {
            if (py::isinstance<py::bool_>(button_object)) {
                throw py::value_error("button must be seat 0 or 1");
            }
            button = button_object.cast<int>();
        }
        if (button < 0 || button >= PLAYERS) {
            throw py::value_error("button must be seat 0 or 1");
        }
        last_button = button;
        state.button = button;
        state.sb_player = button;
        state.bb_player = 1 - button;
        state.small_blind = small_blind;
        state.big_blind = big_blind;

        if (deck_object.is_none()) {
            std::vector<int> deck(52);
            for (int card = 0; card < 52; ++card) deck[card] = card;
            py::list shuffled = py::cast(deck);
            rng.attr("shuffle")(shuffled);
            state.deck = shuffled.cast<std::vector<int>>();
        } else {
            auto deck = deck_object.cast<std::vector<int>>();
            validate_cards(deck, 52);
            state.deck = deck;
        }

        for (int round = 0; round < 2; ++round) {
            // The first card is dealt to the seat left of the dealer. In
            // heads-up that is the big blind, followed by the button/small
            // blind.
            for (int player : {state.bb_player, state.sb_player}) {
                state.hole[player].push_back(state.deck.back());
                state.deck.pop_back();
            }
        }

        state.stacks = state.initial_stacks;
        post_blind(state, state.sb_player, small_blind);
        post_blind(state, state.bb_player, big_blind);
        state.pot = state.total_contrib[0] + state.total_contrib[1];
        state.current_bet = std::max(
            state.street_contrib[0],
            state.street_contrib[1]
        );
        state.min_raise = big_blind;

        for (int player = 0; player < PLAYERS; ++player) {
            state.all_in[player] = state.stacks[player] == 0;
            state.raise_rights[player] = !state.all_in[player];
        }

        const auto active = can_act_mask(state);
        if (std::popcount(active) == PLAYERS) {
            state.pending_mask = active;
            state.to_act = state.sb_player;
        } else {
            const int active_player = active & 1 ? 0 : (active & 2 ? 1 : -1);
            if (
                active_player >= 0
                && state.street_contrib[active_player] < state.current_bet
            ) {
                state.pending_mask = static_cast<std::uint8_t>(1 << active_player);
                state.to_act = active_player;
            } else {
                runout_and_showdown(state);
            }
        }
        assert_chip_conservation(state);
        return state;
    }

    static HeadsUpState clone(const HeadsUpState& state) { return state; }

    Chip amount_to_call(const HeadsUpState& state, int player = -1) const {
        if (player < 0) player = state.to_act;
        if (player < 0) return 0;
        if (player >= PLAYERS) {
            throw py::value_error("player must be seat 0 or 1");
        }
        return std::max<Chip>(
            0,
            state.current_bet - state.street_contrib[player]
        );
    }

    std::vector<ActionOption> action_options(const HeadsUpState& state) const {
        if (state.terminal) return {};
        const int player = state.to_act;
        if (
            player < 0
            || player >= PLAYERS
            || !(state.pending_mask & (1 << player))
        ) {
            throw std::runtime_error(
                "non-terminal state has no valid pending actor"
            );
        }
        if (
            state.folded[player]
            || state.all_in[player]
            || state.stacks[player] <= 0
        ) {
            throw std::runtime_error("folded or all-in player cannot act");
        }

        const Chip contribution = state.street_contrib[player];
        const Chip stack = state.stacks[player];
        const Chip to_call = amount_to_call(state, player);
        std::vector<ActionOption> result;
        std::set<std::pair<int, Chip>> seen_effects;

        auto add = [&](int action,
                       OptionKind kind,
                       std::string semantic,
                       Chip payment,
                       Chip target) {
            const auto effect = std::make_pair(
                static_cast<int>(kind),
                target
            );
            if (seen_effects.insert(effect).second) {
                result.push_back(
                    {action, kind, std::move(semantic), payment, target}
                );
            }
        };

        if (to_call > 0) {
            add(FOLD, OptionKind::Fold, "fold", 0, contribution);
            const Chip payment = std::min(stack, to_call);
            add(
                CALL,
                OptionKind::Commit,
                "call",
                payment,
                contribution + payment
            );
        } else {
            add(CHECK, OptionKind::Check, "check", 0, contribution);
        }

        const int opponent = 1 - player;
        const Chip max_target = contribution + stack;
        const bool opponent_can_respond = !state.folded[opponent]
            && !state.all_in[opponent]
            && state.stacks[opponent] > 0;
        const bool may_raise = state.raise_rights[player]
            && opponent_can_respond
            && max_target > state.current_bet;
        if (!may_raise) {
            sort_options(result);
            return result;
        }

        const Chip minimum_target = state.current_bet + state.min_raise;
        const Chip pot_after_call = state.pot + std::min(stack, to_call);
        const Chip called_target = contribution + std::min(stack, to_call);

        const std::array<std::tuple<int, Chip, Chip>, 5> fractions{{
            {THIRD_POT, 1, 3},
            {HALF_POT, 1, 2},
            {THREE_QUARTER_POT, 3, 4},
            {POT, 1, 1},
            {OVERBET, 3, 2},
        }};

        add_template_raise(
            result,
            seen_effects,
            MIN_RAISE,
            minimum_target,
            contribution,
            minimum_target,
            max_target
        );
        for (const auto& [action, numerator, denominator] : fractions) {
            const Chip target = called_target
                + rounded_fraction(pot_after_call, numerator, denominator);
            add_template_raise(
                result,
                seen_effects,
                action,
                target,
                contribution,
                minimum_target,
                max_target
            );
        }

        const OptionKind all_in_kind = max_target > state.current_bet
            ? OptionKind::Commit
            : OptionKind::Commit;
        const auto all_in_effect = std::make_pair(
            static_cast<int>(all_in_kind),
            max_target
        );
        if (seen_effects.insert(all_in_effect).second) {
            result.push_back(
                {ALL_IN, all_in_kind, "all_in", stack, max_target}
            );
        }
        sort_options(result);
        return result;
    }

    std::vector<int> legal_actions(const HeadsUpState& state) const {
        std::vector<int> actions;
        for (const auto& option : action_options(state)) {
            actions.push_back(option.action);
        }
        return actions;
    }

    std::vector<int> legal_action_mask(const HeadsUpState& state) const {
        std::vector<int> mask(ACTIONS);
        for (int action : legal_actions(state)) mask[action] = 1;
        return mask;
    }

    Chip action_target(const HeadsUpState& state, int action) const {
        for (const auto& option : action_options(state)) {
            if (option.action == action) return option.target;
        }
        throw py::value_error("illegal action");
    }

    Chip action_payment(const HeadsUpState& state, int action) const {
        for (const auto& option : action_options(state)) {
            if (option.action == action) return option.payment;
        }
        throw py::value_error("illegal action");
    }

    py::list action_descriptors(const HeadsUpState& state) const {
        std::array<const ActionOption*, ACTIONS> by_action{};
        const auto options = action_options(state);
        for (const auto& option : options) by_action[option.action] = &option;
        py::list descriptors;
        for (int action = 0; action < ACTIONS; ++action) {
            if (!by_action[action]) {
                descriptors.append(py::none());
                continue;
            }
            const ActionOption& option = *by_action[action];
            const int actor = state.to_act;
            const int opponent = 1 - actor;
            const Chip remaining = state.stacks[actor] - option.payment;
            const bool aggressive = option.target > state.current_bet;
            const bool full_raise = aggressive
                && option.target - state.current_bet >= state.min_raise;
            py::dict descriptor;
            descriptor["action"] = action;
            descriptor["target"] = option.target;
            descriptor["payment"] = option.payment;
            descriptor["resulting_pot"] = state.pot + option.payment;
            descriptor["remaining_stack"] = remaining;
            descriptor["resulting_effective_stack"] = std::min(
                remaining,
                state.stacks[opponent]
            );
            descriptor["is_all_in"] = remaining == 0;
            descriptor["is_aggressive"] = aggressive;
            descriptor["is_full_raise"] = full_raise;
            descriptor["reopens_betting"] = full_raise
                && remaining > 0
                && state.stacks[opponent] > 0;
            descriptors.append(descriptor);
        }
        return descriptors;
    }

    HeadsUpState step(const HeadsUpState& old, int action) {
        if (action < 0 || action >= ACTIONS) {
            throw py::value_error("action must be an integer in 0..9");
        }
        for (const auto& option : action_options(old)) {
            if (option.action == action) {
                return apply_option(old, option, action);
            }
        }
        throw py::value_error(
            "illegal action " + std::to_string(action)
            + " (" + ACTION_NAMES[action] + ")"
        );
    }

    HeadsUpState step_exact(
        const HeadsUpState& old,
        std::string kind,
        py::object raise_to_object = py::none()
    ) {
        if (old.terminal) {
            throw std::runtime_error("cannot act on a terminal state");
        }
        std::transform(
            kind.begin(),
            kind.end(),
            kind.begin(),
            [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            }
        );
        kind.erase(
            kind.begin(),
            std::find_if(
                kind.begin(),
                kind.end(),
                [](unsigned char value) { return !std::isspace(value); }
            )
        );
        kind.erase(
            std::find_if(
                kind.rbegin(),
                kind.rend(),
                [](unsigned char value) { return !std::isspace(value); }
            ).base(),
            kind.end()
        );
        std::replace(kind.begin(), kind.end(), '-', '_');
        if (kind == "fold") {
            ensure_no_raise_to(raise_to_object, kind);
            return apply_option(
                old,
                exact_passive_option(old, OptionKind::Fold),
                -1
            );
        }
        if (kind == "check") {
            ensure_no_raise_to(raise_to_object, kind);
            return apply_option(
                old,
                exact_passive_option(old, OptionKind::Check),
                -1
            );
        }
        if (kind == "call") {
            ensure_no_raise_to(raise_to_object, kind);
            return apply_option(
                old,
                exact_passive_option(old, OptionKind::Commit),
                -1
            );
        }
        if (kind == "all_in" || kind == "allin") {
            ensure_no_raise_to(raise_to_object, kind);
            return apply_option(old, exact_all_in_option(old), -1);
        }
        if (kind == "raise" || kind == "bet" || kind == "raise_to") {
            if (raise_to_object.is_none()) {
                throw py::value_error("raise_to is required for an exact raise");
            }
            if (
                py::isinstance<py::bool_>(raise_to_object)
                || !py::isinstance<py::int_>(raise_to_object)
            ) {
                throw py::value_error(
                    "raise_to must be an integer chip total"
                );
            }
            return apply_option(
                old,
                exact_raise_option(old, raise_to_object.cast<Chip>()),
                -1
            );
        }
        throw py::value_error(
            "kind must be fold, check, call, raise_to, or all_in"
        );
    }

    HeadsUpState resolve_showdown(const HeadsUpState& old) {
        if (old.terminal) {
            throw std::runtime_error("state is already terminal");
        }
        if (old.board.size() != 5) {
            throw py::value_error("showdown requires a five-card board");
        }
        HeadsUpState state = old;
        resolve_showdown_in_place(state);
        return state;
    }

    Chip terminal_payoff(const HeadsUpState& state, int player) const {
        if (player < 0 || player >= PLAYERS) {
            throw py::value_error("player must be seat 0 or 1");
        }
        if (!state.terminal || !state.has_payoffs) {
            throw std::runtime_error(
                "payoff is available only at a terminal state"
            );
        }
        return state.payoffs[player];
    }

private:
    static void sort_options(std::vector<ActionOption>& options) {
        std::sort(
            options.begin(),
            options.end(),
            [](const ActionOption& left, const ActionOption& right) {
                return left.action < right.action;
            }
        );
    }

    static void add_template_raise(
        std::vector<ActionOption>& result,
        std::set<std::pair<int, Chip>>& seen_effects,
        int action,
        Chip target,
        Chip contribution,
        Chip minimum_target,
        Chip maximum_target
    ) {
        if (target < minimum_target || target > maximum_target) return;
        const auto effect = std::make_pair(
            static_cast<int>(OptionKind::Commit),
            target
        );
        if (seen_effects.insert(effect).second) {
            result.push_back(
                {
                    action,
                    OptionKind::Commit,
                    ACTION_NAMES[action],
                    target - contribution,
                    target,
                }
            );
        }
    }

    static std::uint8_t can_act_mask(const HeadsUpState& state) {
        std::uint8_t result = 0;
        for (int player = 0; player < PLAYERS; ++player) {
            if (
                !state.folded[player]
                && !state.all_in[player]
                && state.stacks[player] > 0
            ) {
                result |= static_cast<std::uint8_t>(1 << player);
            }
        }
        return result;
    }

    static int next_clockwise(int start, std::uint8_t mask) {
        for (int distance = 1; distance <= PLAYERS; ++distance) {
            const int player = (start + distance) % PLAYERS;
            if (mask & (1 << player)) return player;
        }
        return -1;
    }

    static void post_blind(
        HeadsUpState& state,
        int player,
        Chip blind
    ) {
        const Chip posted = std::min(state.stacks[player], blind);
        state.stacks[player] -= posted;
        state.total_contrib[player] += posted;
        state.street_contrib[player] += posted;
    }

    ActionOption exact_passive_option(
        const HeadsUpState& state,
        OptionKind requested
    ) const {
        const int player = checked_actor(state);
        const Chip contribution = state.street_contrib[player];
        const Chip to_call = amount_to_call(state, player);
        if (requested == OptionKind::Fold) {
            return {-1, OptionKind::Fold, "fold", 0, contribution};
        }
        if (requested == OptionKind::Check) {
            if (to_call != 0) {
                throw py::value_error("check is illegal when facing a bet");
            }
            return {-1, OptionKind::Check, "check", 0, contribution};
        }
        if (to_call <= 0) {
            throw py::value_error("call is illegal when checking is available");
        }
        const Chip payment = std::min(state.stacks[player], to_call);
        return {
            -1,
            OptionKind::Commit,
            "call",
            payment,
            contribution + payment,
        };
    }

    ActionOption exact_all_in_option(const HeadsUpState& state) const {
        const int player = checked_actor(state);
        const Chip contribution = state.street_contrib[player];
        const Chip stack = state.stacks[player];
        const Chip target = contribution + stack;
        const Chip to_call = amount_to_call(state, player);
        if (target <= state.current_bet) {
            return {
                -1,
                OptionKind::Commit,
                "call",
                std::min(stack, to_call),
                target,
            };
        }
        validate_exact_raise(state, target);
        return {-1, OptionKind::Commit, "all_in", stack, target};
    }

    ActionOption exact_raise_option(
        const HeadsUpState& state,
        Chip target
    ) const {
        const int player = checked_actor(state);
        validate_exact_raise(state, target);
        return {
            -1,
            OptionKind::Commit,
            state.current_bet == 0 ? "bet" : "raise",
            target - state.street_contrib[player],
            target,
        };
    }

    int checked_actor(const HeadsUpState& state) const {
        if (state.terminal) {
            throw std::runtime_error("cannot act on a terminal state");
        }
        const int player = state.to_act;
        if (
            player < 0
            || player >= PLAYERS
            || !(state.pending_mask & (1 << player))
        ) {
            throw std::runtime_error(
                "non-terminal state has no valid pending actor"
            );
        }
        return player;
    }

    void validate_exact_raise(
        const HeadsUpState& state,
        Chip target
    ) const {
        const int player = checked_actor(state);
        const int opponent = 1 - player;
        const Chip contribution = state.street_contrib[player];
        const Chip maximum = contribution + state.stacks[player];
        if (target <= state.current_bet) {
            throw py::value_error(
                "raise_to must exceed the current bet; use call instead"
            );
        }
        if (target <= contribution) {
            throw py::value_error("raise_to must add chips");
        }
        if (target > maximum) {
            throw py::value_error("raise_to exceeds the acting player's stack");
        }
        if (!state.raise_rights[player]) {
            throw py::value_error("raising has not been reopened");
        }
        if (
            state.folded[opponent]
            || state.all_in[opponent]
            || state.stacks[opponent] <= 0
        ) {
            throw py::value_error("cannot raise when the opponent cannot respond");
        }
        const Chip minimum = state.current_bet + state.min_raise;
        if (target < minimum && target != maximum) {
            throw py::value_error(
                "a sub-minimum raise is legal only when it is all-in"
            );
        }
    }

    static void ensure_no_raise_to(
        const py::object& value,
        const std::string& kind
    ) {
        if (!value.is_none()) {
            throw py::value_error(
                "raise_to is only valid with an exact raise, not " + kind
            );
        }
    }

    HeadsUpState apply_option(
        const HeadsUpState& old,
        const ActionOption& option,
        int recorded_action
    ) {
        if (old.terminal) {
            throw std::runtime_error("cannot act on a terminal state");
        }
        HeadsUpState state = old;
        const int player = checked_actor(state);
        const Chip current_before = state.current_bet;
        const Chip pot_before = state.pot;
        const Chip to_call_before = amount_to_call(state, player);
        bool full_raise = false;

        if (option.kind == OptionKind::Fold) {
            state.folded[player] = true;
            state.raise_rights[player] = false;
            state.has_last_action_bet[player] = true;
            state.last_action_bet[player] = state.current_bet;
            state.pending_mask &= static_cast<std::uint8_t>(~(1 << player));
        } else if (option.kind == OptionKind::Check) {
            state.raise_rights[player] = false;
            state.has_last_action_bet[player] = true;
            state.last_action_bet[player] = state.current_bet;
            state.pending_mask &= static_cast<std::uint8_t>(~(1 << player));
        } else {
            if (option.payment < 0 || option.payment > state.stacks[player]) {
                throw std::runtime_error(
                    "internal action target exceeds the acting stack"
                );
            }
            state.stacks[player] -= option.payment;
            state.street_contrib[player] += option.payment;
            state.total_contrib[player] += option.payment;
            state.pot += option.payment;
            if (state.stacks[player] == 0) state.all_in[player] = true;
            const Chip new_total = state.street_contrib[player];

            state.pending_mask &= static_cast<std::uint8_t>(~(1 << player));
            if (new_total > current_before) {
                const Chip increment = new_total - current_before;
                const Chip old_min_raise = state.min_raise;
                full_raise = increment >= old_min_raise;
                state.current_bet = new_total;
                if (full_raise) {
                    state.min_raise = increment;
                    state.last_full_raiser = player;
                    const int opponent = 1 - player;
                    if (can_act_mask(state) & (1 << opponent)) {
                        state.raise_rights[opponent] = true;
                    }
                }

                state.pending_mask = 0;
                const int opponent = 1 - player;
                if (
                    (can_act_mask(state) & (1 << opponent))
                    && state.street_contrib[opponent] < state.current_bet
                ) {
                    state.pending_mask |= static_cast<std::uint8_t>(
                        1 << opponent
                    );
                }
            }
            state.raise_rights[player] = false;
            state.has_last_action_bet[player] = true;
            state.last_action_bet[player] = state.current_bet;
        }

        std::string semantic;
        if (option.kind == OptionKind::Fold) {
            semantic = "fold";
        } else if (option.kind == OptionKind::Check) {
            semantic = "check";
        } else if (state.current_bet > current_before) {
            semantic = current_before == 0 ? "bet" : "raise";
        } else {
            semantic = "call";
        }
        append_history(
            state,
            {
                player,
                state.street,
                recorded_action,
                semantic,
                option.payment,
                option.target,
                state.street_contrib[player],
                current_before,
                state.current_bet,
                pot_before,
                state.pot,
                to_call_before,
                full_raise,
                state.all_in[player],
            }
        );

        const int opponent = 1 - player;
        if (state.folded[player]) {
            award_uncontested(state, opponent);
            return state;
        }
        if (state.folded[opponent]) {
            award_uncontested(state, player);
            return state;
        }

        const auto active = can_act_mask(state);
        state.pending_mask &= active;
        if (std::popcount(active) < PLAYERS) {
            bool contributions_matched = true;
            for (int seat = 0; seat < PLAYERS; ++seat) {
                if (
                    (active & (1 << seat))
                    && state.street_contrib[seat] < state.current_bet
                ) {
                    contributions_matched = false;
                }
            }
            if (contributions_matched) state.pending_mask = 0;
        }
        if (state.pending_mask == 0) {
            close_betting_round(state);
        } else {
            state.to_act = next_clockwise(player, state.pending_mask);
        }
        assert_chip_conservation(state);
        return state;
    }

    void close_betting_round(HeadsUpState& state) {
        if (state.street == RIVER) {
            resolve_showdown_in_place(state);
            return;
        }
        if (std::popcount(can_act_mask(state)) < PLAYERS) {
            runout_and_showdown(state);
            return;
        }
        advance_street(state);
    }

    void advance_street(HeadsUpState& state) {
        if (state.street == PREFLOP) {
            burn(state);
            deal_board(state, 3);
            state.street = FLOP;
        } else if (state.street == FLOP) {
            burn(state);
            deal_board(state, 1);
            state.street = TURN;
        } else if (state.street == TURN) {
            burn(state);
            deal_board(state, 1);
            state.street = RIVER;
        } else {
            throw std::runtime_error("cannot advance beyond the river");
        }
        state.street_contrib.fill(0);
        state.current_bet = 0;
        state.min_raise = big_blind;
        state.last_full_raiser = -1;
        state.last_action_bet.fill(0);
        state.has_last_action_bet.fill(false);
        const auto active = can_act_mask(state);
        state.pending_mask = active;
        for (int player = 0; player < PLAYERS; ++player) {
            state.raise_rights[player] = active & (1 << player);
        }
        state.to_act = state.bb_player;
    }

    void runout_and_showdown(HeadsUpState& state) {
        while (state.board.size() < 5) {
            if (state.board.empty()) {
                burn(state);
                deal_board(state, 3);
                state.street = FLOP;
            } else if (state.board.size() == 3) {
                burn(state);
                deal_board(state, 1);
                state.street = TURN;
            } else if (state.board.size() == 4) {
                burn(state);
                deal_board(state, 1);
                state.street = RIVER;
            } else {
                throw std::runtime_error("board has an invalid number of cards");
            }
        }
        state.street = RIVER;
        resolve_showdown_in_place(state);
    }

    static void burn(HeadsUpState& state) {
        if (state.deck.empty()) {
            throw std::runtime_error("deck exhausted while burning");
        }
        state.burned.push_back(state.deck.back());
        state.deck.pop_back();
    }

    static void deal_board(HeadsUpState& state, int count) {
        if (static_cast<int>(state.deck.size()) < count) {
            throw std::runtime_error("deck exhausted while dealing board");
        }
        while (count-- > 0) {
            state.board.push_back(state.deck.back());
            state.deck.pop_back();
        }
    }

    static void refund_uncalled(HeadsUpState& state) {
        if (state.total_contrib[0] == state.total_contrib[1]) return;
        const int player = state.total_contrib[0] > state.total_contrib[1] ? 0 : 1;
        const Chip refund = state.total_contrib[player]
            - state.total_contrib[1 - player];
        if (refund <= 0 || refund > state.pot) {
            throw std::runtime_error("invalid uncalled-bet refund");
        }
        if (refund > state.street_contrib[player]) {
            throw std::runtime_error(
                "uncalled excess is not on the current street"
            );
        }
        state.total_contrib[player] -= refund;
        state.street_contrib[player] -= refund;
        state.current_bet = std::max(
            state.street_contrib[0],
            state.street_contrib[1]
        );
        state.stacks[player] += refund;
        state.pot -= refund;
        state.uncalled_returns[player] += refund;
    }

    static void resolve_showdown_in_place(HeadsUpState& state) {
        if (state.board.size() != 5) {
            throw std::runtime_error("showdown requires five board cards");
        }
        std::vector<int> all_cards;
        for (int player = 0; player < PLAYERS; ++player) {
            all_cards.insert(
                all_cards.end(),
                state.hole[player].begin(),
                state.hole[player].end()
            );
        }
        all_cards.insert(
            all_cards.end(),
            state.board.begin(),
            state.board.end()
        );
        validate_cards(all_cards, 9);
        refund_uncalled(state);
        std::array<int, PLAYERS> scores{};
        for (int player = 0; player < PLAYERS; ++player) {
            std::vector<int> cards = state.hole[player].vector();
            cards.insert(cards.end(), state.board.begin(), state.board.end());
            scores[player] = evaluate_7card(cards);
        }
        std::array<Chip, PLAYERS> awards{};
        std::vector<int> winners;
        if (scores[0] > scores[1]) {
            awards[0] = state.pot;
            winners.push_back(0);
        } else if (scores[1] > scores[0]) {
            awards[1] = state.pot;
            winners.push_back(1);
        } else {
            awards[0] = state.pot / 2;
            awards[1] = state.pot / 2;
            if (state.pot % 2 != 0) {
                ++awards[state.bb_player];
            }
            winners = {0, 1};
        }
        for (int player = 0; player < PLAYERS; ++player) {
            state.stacks[player] += awards[player];
        }
        finish(state, awards, winners);
    }

    static void award_uncontested(HeadsUpState& state, int winner) {
        refund_uncalled(state);
        std::array<Chip, PLAYERS> awards{};
        awards[winner] = state.pot;
        state.stacks[winner] += state.pot;
        finish(state, awards, {winner});
    }

    static void finish(
        HeadsUpState& state,
        const std::array<Chip, PLAYERS>& awards,
        const std::vector<int>& winners
    ) {
        state.pot = 0;
        state.terminal = true;
        state.to_act = -1;
        state.pending_mask = 0;
        state.payouts = awards;
        state.has_payouts = true;
        state.winners = winners;
        for (int player = 0; player < PLAYERS; ++player) {
            state.payoffs[player] = state.stacks[player]
                - state.initial_stacks[player];
        }
        state.has_payoffs = true;
        assert_chip_conservation(state);
    }

    static void assert_chip_conservation(const HeadsUpState& state) {
        const Chip expected = state.initial_stacks[0] + state.initial_stacks[1];
        const Chip actual = state.stacks[0] + state.stacks[1] + state.pot;
        if (actual != expected) {
            throw std::runtime_error("chip conservation failed");
        }
        if (
            state.has_payoffs
            && state.payoffs[0] + state.payoffs[1] != 0
        ) {
            throw std::runtime_error("terminal payoffs are not zero-sum");
        }
    }
};

constexpr int CARD_FEATURES = 18;
constexpr int CARD_TOKEN_COUNT = 7;
constexpr int PUBLIC_PREFIX_FEATURES = 56;
constexpr int HISTORY_FEATURES = 23;
constexpr int ACTION_DESCRIPTOR_FEATURES = 11;

static void append_one_hot(
    std::vector<float>& output,
    int selected,
    int width
) {
    for (int index = 0; index < width; ++index) {
        output.push_back(selected == index ? 1.0f : 0.0f);
    }
}

static void append_card_features(std::vector<float>& output, int card) {
    const auto start = output.size();
    output.resize(start + CARD_FEATURES, 0.0f);
    if (card < 0) return;
    if (card >= 52) {
        throw py::value_error("card index must be in [0, 51]");
    }
    output[start + static_cast<std::size_t>(card % 13)] = 1.0f;
    output[start + static_cast<std::size_t>(13 + card / 13)] = 1.0f;
    output[start + 17] = 1.0f;
}

static std::string normalized_kind(const ActionRecord& event) {
    std::string kind = event.kind;
    std::transform(
        kind.begin(),
        kind.end(),
        kind.begin(),
        [](unsigned char value) {
            if (value == '-' || value == ' ') return '_';
            return static_cast<char>(std::tolower(value));
        }
    );
    if (
        kind == "fold"
        || kind == "check"
        || kind == "call"
        || kind == "bet"
        || kind == "raise"
    ) {
        return kind;
    }
    if (event.current_bet_after > event.current_bet_before) {
        return event.current_bet_before == 0 ? "bet" : "raise";
    }
    if (event.amount > 0) return "call";
    return "check";
}

static py::object descriptor_field(
    const py::handle& descriptor,
    const char* name
) {
    if (py::isinstance<py::dict>(descriptor)) {
        const auto mapping = py::reinterpret_borrow<py::dict>(descriptor);
        if (!mapping.contains(name)) {
            throw py::value_error(
                std::string("action descriptor is missing '") + name + "'"
            );
        }
        return py::reinterpret_borrow<py::object>(mapping[name]);
    }
    if (!py::hasattr(descriptor, name)) {
        throw py::value_error(
            std::string("action descriptor is missing '") + name + "'"
        );
    }
    return py::getattr(descriptor, name);
}

static std::vector<py::object> descriptors_by_action(
    const py::object& descriptors_object
) {
    std::vector<py::object> result(
        static_cast<std::size_t>(ACTIONS),
        py::none()
    );
    if (descriptors_object.is_none()) return result;

    if (py::isinstance<py::dict>(descriptors_object)) {
        const auto mapping = descriptors_object.cast<py::dict>();
        for (int action = 0; action < ACTIONS; ++action) {
            const auto key = py::int_(action);
            if (mapping.contains(key)) {
                result[static_cast<std::size_t>(action)] =
                    py::reinterpret_borrow<py::object>(mapping[key]);
            }
        }
        return result;
    }

    const auto sequence = descriptors_object.cast<py::sequence>();
    if (py::len(sequence) != ACTIONS) {
        throw py::value_error("action_descriptors must contain 10 entries");
    }
    for (int action = 0; action < ACTIONS; ++action) {
        result[static_cast<std::size_t>(action)] =
            py::reinterpret_borrow<py::object>(sequence[action]);
    }
    return result;
}

static py::array_t<float> encode_information_state_native(
    const HeadsUpState& state,
    int hero,
    const std::vector<int>& legal_actions,
    double big_blind,
    int max_history,
    py::object action_descriptors_object
) {
    if (hero < 0 || hero >= PLAYERS) {
        throw py::value_error("hero must be seat 0 or 1");
    }
    if (!(big_blind > 0.0)) {
        throw py::value_error("big_blind must be positive");
    }
    if (max_history <= 0) {
        throw py::value_error("max_history must be a positive integer");
    }
    if (!legal_actions.empty() && state.to_act != hero) {
        throw py::value_error(
            "live decision encoding requires hero == state.to_act so legal "
            "actions and exact descriptors belong to the encoded player"
        );
    }
    const int expected = PUBLIC_PREFIX_FEATURES
        + CARD_TOKEN_COUNT * CARD_FEATURES
        + max_history * HISTORY_FEATURES
        + ACTIONS
        + ACTIONS * ACTION_DESCRIPTOR_FEATURES;
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(expected));
    const int opponent = 1 - hero;
    const double bb = big_blind;

    append_one_hot(values, state.street, 4);
    append_one_hot(values, (state.button - hero + PLAYERS) % PLAYERS, PLAYERS);
    values.push_back(hero == state.button ? 1.0f : 0.0f);
    values.push_back(hero == state.sb_player ? 1.0f : 0.0f);
    values.push_back(hero == state.bb_player ? 1.0f : 0.0f);
    values.push_back(state.to_act < 0 ? 1.0f : 0.0f);
    values.push_back(state.to_act == hero ? 1.0f : 0.0f);
    values.push_back(state.to_act == opponent ? 1.0f : 0.0f);
    values.push_back(state.last_full_raiser < 0 ? 1.0f : 0.0f);
    values.push_back(state.last_full_raiser == hero ? 1.0f : 0.0f);
    values.push_back(state.last_full_raiser == opponent ? 1.0f : 0.0f);

    for (int seat : {hero, opponent}) {
        values.push_back(static_cast<float>(state.stacks[seat] / bb));
        values.push_back(
            static_cast<float>(state.initial_stacks[seat] / bb)
        );
        values.push_back(
            static_cast<float>(state.total_contrib[seat] / bb)
        );
        values.push_back(
            static_cast<float>(state.street_contrib[seat] / bb)
        );
        values.push_back(state.folded[seat] ? 1.0f : 0.0f);
        values.push_back(state.all_in[seat] ? 1.0f : 0.0f);
        values.push_back(
            state.pending_mask & (1 << seat) ? 1.0f : 0.0f
        );
        values.push_back(state.raise_rights[seat] ? 1.0f : 0.0f);
        values.push_back(
            state.has_last_action_bet[seat]
            ? static_cast<float>(state.last_action_bet[seat] / bb)
            : 0.0f
        );
        values.push_back(
            state.has_last_action_bet[seat] ? 1.0f : 0.0f
        );
    }

    const double pot = static_cast<double>(state.pot);
    const double current_bet = static_cast<double>(state.current_bet);
    const double minimum_raise = static_cast<double>(state.min_raise);
    const double minimum_raise_to = current_bet + minimum_raise;
    const double to_call = std::max(
        0.0,
        current_bet - static_cast<double>(state.street_contrib[hero])
    );
    const double call_payment = std::min(
        static_cast<double>(state.stacks[hero]),
        to_call
    );
    const double maximum_raise_to = static_cast<double>(
        state.street_contrib[hero] + state.stacks[hero]
    );
    const double pot_after_call = pot + call_payment;
    const double effective_stack = static_cast<double>(
        std::min(state.stacks[hero], state.stacks[opponent])
    );
    const double hero_after_call = std::max(
        0.0,
        static_cast<double>(state.stacks[hero]) - call_payment
    );
    const double effective_after_call = std::min(
        hero_after_call,
        static_cast<double>(state.stacks[opponent])
    );
    const int active_count = static_cast<int>(!state.folded[0])
        + static_cast<int>(!state.folded[1]);
    const int pending_count = std::popcount(state.pending_mask);
    const double small_blind = state.small_blind > 0
        ? static_cast<double>(state.small_blind)
        : 0.5 * bb;

    const std::array<double, 21> globals{{
        pot / bb,
        current_bet / bb,
        minimum_raise / bb,
        minimum_raise_to / bb,
        to_call / bb,
        call_payment / bb,
        maximum_raise_to / bb,
        pot_after_call / bb,
        effective_stack / bb,
        effective_after_call / bb,
        pot_after_call > 1e-9 ? effective_after_call / pot_after_call : 0.0,
        pot_after_call > 1e-9 ? call_payment / pot_after_call : 0.0,
        pot > 1e-9 ? current_bet / pot : 0.0,
        pot_after_call > 1e-9 ? minimum_raise_to / pot_after_call : 0.0,
        pot_after_call > 1e-9 ? maximum_raise_to / pot_after_call : 0.0,
        static_cast<double>(state.board.size()) / 5.0,
        static_cast<double>(active_count) / 2.0,
        static_cast<double>(pending_count) / 2.0,
        state.raise_rights[hero] ? 1.0 : 0.0,
        small_blind / bb,
        static_cast<double>(
            state.initial_stacks[0] + state.initial_stacks[1]
        ) / bb,
    }};
    for (double value : globals) values.push_back(static_cast<float>(value));

    if (state.hole[hero].size() != 2) {
        throw py::value_error("hero must have exactly two hole cards");
    }
    std::array<int, 2> hero_cards{{
        state.hole[hero][0],
        state.hole[hero][1],
    }};
    std::sort(hero_cards.begin(), hero_cards.end());
    for (int card : hero_cards) append_card_features(values, card);
    std::vector<int> canonical_board = state.board.vector();
    if (canonical_board.size() >= 3) {
        std::sort(canonical_board.begin(), canonical_board.begin() + 3);
    }
    for (int index = 0; index < 5; ++index) {
        append_card_features(
            values,
            index < static_cast<int>(canonical_board.size())
                ? canonical_board[static_cast<std::size_t>(index)]
                : -1
        );
    }

    const auto full_history = history_vector(state);
    const int used_history = std::min(
        max_history,
        static_cast<int>(full_history.size())
    );
    values.insert(
        values.end(),
        static_cast<std::size_t>(
            (max_history - used_history) * HISTORY_FEATURES
        ),
        0.0f
    );
    const auto history_start = full_history.end() - used_history;
    for (auto iterator = history_start; iterator != full_history.end(); ++iterator) {
        const ActionRecord& event = *iterator;
        const std::string kind = normalized_kind(event);
        values.push_back(1.0f);
        append_one_hot(values, event.street, 4);
        values.push_back(event.player == hero ? 1.0f : 0.0f);
        values.push_back(event.player == opponent ? 1.0f : 0.0f);
        for (const char* semantic :
             {"fold", "check", "call", "bet", "raise"}) {
            values.push_back(kind == semantic ? 1.0f : 0.0f);
        }
        values.push_back(event.all_in ? 1.0f : 0.0f);
        values.push_back(event.full_raise ? 1.0f : 0.0f);
        const double amount = static_cast<double>(event.amount);
        const double contribution_after = static_cast<double>(
            event.contribution_after
        );
        const double current_before = static_cast<double>(
            event.current_bet_before
        );
        const double current_after = static_cast<double>(
            event.current_bet_after
        );
        const double raise_increment = std::max(
            0.0,
            current_after - current_before
        );
        const double event_pot_before = static_cast<double>(event.pot_before);
        const double event_pot_after = static_cast<double>(event.pot_after);
        for (double raw : {
             amount,
             contribution_after,
             current_before,
             current_after,
             raise_increment,
             event_pot_before,
             event_pot_after}) {
            values.push_back(static_cast<float>(raw / bb));
        }
        values.push_back(
            static_cast<float>(
                event_pot_before > 1e-9 ? amount / event_pot_before : 0.0
            )
        );
        values.push_back(
            static_cast<float>(
                event_pot_before > 1e-9
                ? contribution_after / event_pot_before
                : 0.0
            )
        );
    }

    std::array<bool, ACTIONS> legal{};
    for (int action : legal_actions) {
        if (action < 0 || action >= ACTIONS) {
            throw py::value_error("legal action is outside valid range");
        }
        legal[action] = true;
    }
    for (bool is_legal : legal) {
        values.push_back(is_legal ? 1.0f : 0.0f);
    }

    if (action_descriptors_object.is_none() && !legal_actions.empty()) {
        throw py::value_error(
            "action_descriptors are required for a live state with legal actions"
        );
    }
    const auto descriptors = descriptors_by_action(action_descriptors_object);
    if (!action_descriptors_object.is_none()) {
        for (int action = 0; action < ACTIONS; ++action) {
            const bool supplied = !descriptors[action].is_none();
            if (legal[action] && !supplied) {
                throw py::value_error(
                    "missing descriptor for legal action "
                    + std::to_string(action)
                );
            }
            if (!legal[action] && supplied) {
                throw py::value_error(
                    "descriptor supplied for illegal action "
                    + std::to_string(action)
                );
            }
        }
    }

    for (int action = 0; action < ACTIONS; ++action) {
        const py::object& descriptor = descriptors[action];
        if (descriptor.is_none()) {
            values.insert(
                values.end(),
                ACTION_DESCRIPTOR_FEATURES,
                0.0f
            );
            continue;
        }
        const double target = descriptor_field(
            descriptor,
            "target"
        ).cast<double>();
        const double payment = descriptor_field(
            descriptor,
            "payment"
        ).cast<double>();
        const double resulting_pot = descriptor_field(
            descriptor,
            "resulting_pot"
        ).cast<double>();
        const double remaining = descriptor_field(
            descriptor,
            "remaining_stack"
        ).cast<double>();
        const double resulting_effective = descriptor_field(
            descriptor,
            "resulting_effective_stack"
        ).cast<double>();
        const std::array<double, ACTION_DESCRIPTOR_FEATURES> encoded{{
            payment / bb,
            target / bb,
            resulting_pot / bb,
            pot_after_call > 1e-9 ? payment / pot_after_call : 0.0,
            pot_after_call > 1e-9 ? target / pot_after_call : 0.0,
            remaining / bb,
            resulting_pot > 1e-9
                ? resulting_effective / resulting_pot
                : 0.0,
            descriptor_field(descriptor, "is_all_in").cast<bool>()
                ? 1.0 : 0.0,
            descriptor_field(descriptor, "is_aggressive").cast<bool>()
                ? 1.0 : 0.0,
            descriptor_field(descriptor, "is_full_raise").cast<bool>()
                ? 1.0 : 0.0,
            descriptor_field(descriptor, "reopens_betting").cast<bool>()
                ? 1.0 : 0.0,
        }};
        for (double value : encoded) values.push_back(static_cast<float>(value));
    }

    if (static_cast<int>(values.size()) != expected) {
        throw std::runtime_error(
            "native encoder produced "
            + std::to_string(values.size())
            + " values; expected "
            + std::to_string(expected)
        );
    }
    py::array_t<float> output(expected);
    auto destination = output.mutable_unchecked<1>();
    for (int index = 0; index < expected; ++index) {
        destination(index) = values[static_cast<std::size_t>(index)];
    }
    return output;
}

template <typename T>
static std::vector<T> array_vector(const std::array<T, PLAYERS>& values) {
    return {values[0], values[1]};
}

template <typename T>
static void assign_two(
    std::array<T, PLAYERS>& destination,
    const std::vector<T>& values
) {
    if (values.size() != PLAYERS) {
        throw py::value_error("field needs two values");
    }
    std::copy(values.begin(), values.end(), destination.begin());
}

static py::dict state_to_dict(const HeadsUpState& state) {
    py::dict result;
    result["deck"] = state.deck.vector();
    result["board"] = state.board.vector();
    result["burned"] = state.burned.vector();
    std::vector<std::vector<int>> hole;
    for (const auto& cards : state.hole) hole.push_back(cards.vector());
    result["hole"] = hole;
    result["stacks"] = array_vector(state.stacks);
    result["initial_stacks"] = array_vector(state.initial_stacks);
    result["total_contrib"] = array_vector(state.total_contrib);
    result["street_contrib"] = array_vector(state.street_contrib);
    result["folded"] = array_vector(state.folded);
    result["all_in"] = array_vector(state.all_in);
    result["raise_rights"] = array_vector(state.raise_rights);
    result["last_action_bet"] = array_vector(state.last_action_bet);
    result["has_last_action_bet"] = array_vector(state.has_last_action_bet);
    result["uncalled_returns"] = array_vector(state.uncalled_returns);
    result["small_blind"] = state.small_blind;
    result["big_blind"] = state.big_blind;
    result["pot"] = state.pot;
    result["current_bet"] = state.current_bet;
    result["min_raise"] = state.min_raise;
    result["to_act"] = state.to_act;
    result["street"] = state.street;
    result["button"] = state.button;
    result["sb_player"] = state.sb_player;
    result["bb_player"] = state.bb_player;
    result["last_full_raiser"] = state.last_full_raiser;
    result["pending_mask"] = state.pending_mask;
    result["terminal"] = state.terminal;
    result["has_payoffs"] = state.has_payoffs;
    result["has_payouts"] = state.has_payouts;
    result["payoffs"] = array_vector(state.payoffs);
    result["payouts"] = array_vector(state.payouts);
    result["winners"] = state.winners.vector();
    py::list history;
    for (const auto& event : history_vector(state)) {
        history.append(
            py::make_tuple(
                event.player,
                event.street,
                event.action,
                event.kind,
                event.amount,
                event.raise_to,
                event.contribution_after,
                event.current_bet_before,
                event.current_bet_after,
                event.pot_before,
                event.pot_after,
                event.to_call_before,
                event.full_raise,
                event.all_in
            )
        );
    }
    result["history"] = history;
    return result;
}

static HeadsUpState state_from_dict(const py::dict& payload) {
    HeadsUpState state;
    state.deck = payload["deck"].cast<std::vector<int>>();
    state.board = payload["board"].cast<std::vector<int>>();
    state.burned = payload["burned"].cast<std::vector<int>>();
    const auto hole = payload["hole"].cast<std::vector<std::vector<int>>>();
    if (hole.size() != PLAYERS) throw py::value_error("invalid native state");
    for (int player = 0; player < PLAYERS; ++player) {
        state.hole[player] = hole[player];
    }
    assign_two(
        state.stacks,
        payload["stacks"].cast<std::vector<Chip>>()
    );
    assign_two(
        state.initial_stacks,
        payload["initial_stacks"].cast<std::vector<Chip>>()
    );
    assign_two(
        state.total_contrib,
        payload["total_contrib"].cast<std::vector<Chip>>()
    );
    assign_two(
        state.street_contrib,
        payload["street_contrib"].cast<std::vector<Chip>>()
    );
    assign_two(
        state.folded,
        payload["folded"].cast<std::vector<bool>>()
    );
    assign_two(
        state.all_in,
        payload["all_in"].cast<std::vector<bool>>()
    );
    assign_two(
        state.raise_rights,
        payload["raise_rights"].cast<std::vector<bool>>()
    );
    assign_two(
        state.last_action_bet,
        payload["last_action_bet"].cast<std::vector<Chip>>()
    );
    assign_two(
        state.has_last_action_bet,
        payload["has_last_action_bet"].cast<std::vector<bool>>()
    );
    assign_two(
        state.uncalled_returns,
        payload["uncalled_returns"].cast<std::vector<Chip>>()
    );
    state.small_blind = payload["small_blind"].cast<Chip>();
    state.big_blind = payload["big_blind"].cast<Chip>();
    state.pot = payload["pot"].cast<Chip>();
    state.current_bet = payload["current_bet"].cast<Chip>();
    state.min_raise = payload["min_raise"].cast<Chip>();
    state.to_act = payload["to_act"].cast<int>();
    state.street = payload["street"].cast<int>();
    state.button = payload["button"].cast<int>();
    state.sb_player = payload["sb_player"].cast<int>();
    state.bb_player = payload["bb_player"].cast<int>();
    state.last_full_raiser = payload["last_full_raiser"].cast<int>();
    state.pending_mask = payload["pending_mask"].cast<std::uint8_t>();
    state.terminal = payload["terminal"].cast<bool>();
    state.has_payoffs = payload["has_payoffs"].cast<bool>();
    state.has_payouts = payload["has_payouts"].cast<bool>();
    assign_two(
        state.payoffs,
        payload["payoffs"].cast<std::vector<Chip>>()
    );
    assign_two(
        state.payouts,
        payload["payouts"].cast<std::vector<Chip>>()
    );
    state.winners = payload["winners"].cast<std::vector<int>>();
    for (py::handle item : payload["history"].cast<py::list>()) {
        const auto tuple = py::reinterpret_borrow<py::tuple>(item);
        append_history(
            state,
            {
                tuple[0].cast<int>(),
                tuple[1].cast<int>(),
                tuple[2].cast<int>(),
                tuple[3].cast<std::string>(),
                tuple[4].cast<Chip>(),
                tuple[5].cast<Chip>(),
                tuple[6].cast<Chip>(),
                tuple[7].cast<Chip>(),
                tuple[8].cast<Chip>(),
                tuple[9].cast<Chip>(),
                tuple[10].cast<Chip>(),
                tuple[11].cast<Chip>(),
                tuple[12].cast<bool>(),
                tuple[13].cast<bool>(),
            }
        );
    }
    return state;
}

PYBIND11_MODULE(heads_up_native_engine, module) {
    module.doc() = "Integer-chip native heads-up no-limit Hold'em engine";

    py::class_<ActionRecord>(module, "ActionRecord", py::module_local())
        .def_readonly("player", &ActionRecord::player)
        .def_readonly("street", &ActionRecord::street)
        .def_property_readonly(
            "action",
            [](const ActionRecord& event) -> py::object {
                return event.action < 0 ? py::none() : py::cast(event.action);
            }
        )
        .def_readonly("kind", &ActionRecord::kind)
        .def_property_readonly(
            "action_name",
            [](const ActionRecord& event) { return event.kind; }
        )
        .def_readonly("amount", &ActionRecord::amount)
        .def_readonly("amount_added", &ActionRecord::amount)
        .def_readonly("raise_to", &ActionRecord::raise_to)
        .def_readonly("target", &ActionRecord::raise_to)
        .def_readonly("contribution_after", &ActionRecord::contribution_after)
        .def_readonly("current_bet_before", &ActionRecord::current_bet_before)
        .def_readonly("current_bet_after", &ActionRecord::current_bet_after)
        .def_readonly("pot_before", &ActionRecord::pot_before)
        .def_readonly("pot_after", &ActionRecord::pot_after)
        .def_readonly("to_call_before", &ActionRecord::to_call_before)
        .def_readonly("full_raise", &ActionRecord::full_raise)
        .def_readonly("all_in", &ActionRecord::all_in);

    py::class_<HeadsUpState>(module, "HeadsUpState")
        .def(py::init<>())
        .def_property(
            "deck",
            [](const HeadsUpState& state) { return state.deck.vector(); },
            [](HeadsUpState& state, const std::vector<int>& values) {
                state.deck = values;
            }
        )
        .def_property(
            "board",
            [](const HeadsUpState& state) { return state.board.vector(); },
            [](HeadsUpState& state, const std::vector<int>& values) {
                state.board = values;
            }
        )
        .def_property(
            "burned",
            [](const HeadsUpState& state) { return state.burned.vector(); },
            [](HeadsUpState& state, const std::vector<int>& values) {
                state.burned = values;
            }
        )
        .def_property(
            "hole",
            [](const HeadsUpState& state) {
                std::vector<std::vector<int>> values;
                for (const auto& cards : state.hole) {
                    values.push_back(cards.vector());
                }
                return values;
            },
            [](HeadsUpState& state,
               const std::vector<std::vector<int>>& values) {
                if (values.size() != PLAYERS) {
                    throw py::value_error("hole needs two seats");
                }
                for (int player = 0; player < PLAYERS; ++player) {
                    state.hole[player] = values[player];
                }
            }
        )
        .def_property(
            "stacks",
            [](const HeadsUpState& state) {
                return array_vector(state.stacks);
            },
            [](HeadsUpState& state, const std::vector<Chip>& values) {
                assign_two(state.stacks, values);
            }
        )
        .def_property(
            "initial_stacks",
            [](const HeadsUpState& state) {
                return array_vector(state.initial_stacks);
            },
            [](HeadsUpState& state, const std::vector<Chip>& values) {
                assign_two(state.initial_stacks, values);
            }
        )
        .def_property(
            "total_contrib",
            [](const HeadsUpState& state) {
                return array_vector(state.total_contrib);
            },
            [](HeadsUpState& state, const std::vector<Chip>& values) {
                assign_two(state.total_contrib, values);
            }
        )
        .def_property(
            "street_contrib",
            [](const HeadsUpState& state) {
                return array_vector(state.street_contrib);
            },
            [](HeadsUpState& state, const std::vector<Chip>& values) {
                assign_two(state.street_contrib, values);
            }
        )
        .def_property(
            "folded",
            [](const HeadsUpState& state) {
                return array_vector(state.folded);
            },
            [](HeadsUpState& state, const std::vector<bool>& values) {
                assign_two(state.folded, values);
            }
        )
        .def_property(
            "all_in",
            [](const HeadsUpState& state) {
                return array_vector(state.all_in);
            },
            [](HeadsUpState& state, const std::vector<bool>& values) {
                assign_two(state.all_in, values);
            }
        )
        .def_property(
            "raise_rights",
            [](const HeadsUpState& state) {
                return array_vector(state.raise_rights);
            },
            [](HeadsUpState& state, const std::vector<bool>& values) {
                assign_two(state.raise_rights, values);
            }
        )
        .def_property_readonly(
            "last_action_bet",
            [](const HeadsUpState& state) {
                py::list values;
                for (int player = 0; player < PLAYERS; ++player) {
                    values.append(
                        state.has_last_action_bet[player]
                        ? py::cast(state.last_action_bet[player])
                        : py::none()
                    );
                }
                return values;
            }
        )
        .def_property_readonly(
            "uncalled_returns",
            [](const HeadsUpState& state) {
                return array_vector(state.uncalled_returns);
            }
        )
        .def_readonly("small_blind", &HeadsUpState::small_blind)
        .def_readonly("big_blind", &HeadsUpState::big_blind)
        .def_property_readonly(
            "sb",
            [](const HeadsUpState& state) { return state.small_blind; }
        )
        .def_property_readonly(
            "bb",
            [](const HeadsUpState& state) { return state.big_blind; }
        )
        .def_readwrite("pot", &HeadsUpState::pot)
        .def_readwrite("current_bet", &HeadsUpState::current_bet)
        .def_readwrite("min_raise", &HeadsUpState::min_raise)
        .def_property_readonly(
            "to_act",
            [](const HeadsUpState& state) -> py::object {
                return state.to_act < 0 ? py::none() : py::cast(state.to_act);
            }
        )
        .def_readwrite("street", &HeadsUpState::street)
        .def_readwrite("button", &HeadsUpState::button)
        .def_readwrite("sb_player", &HeadsUpState::sb_player)
        .def_readwrite("bb_player", &HeadsUpState::bb_player)
        .def_property_readonly(
            "last_full_raiser",
            [](const HeadsUpState& state) -> py::object {
                return state.last_full_raiser < 0
                    ? py::none()
                    : py::cast(state.last_full_raiser);
            }
        )
        .def_property_readonly(
            "pending_actors",
            [](const HeadsUpState& state) {
                py::set values;
                for (int player = 0; player < PLAYERS; ++player) {
                    if (state.pending_mask & (1 << player)) values.add(player);
                }
                return values;
            }
        )
        .def_property_readonly(
            "pending",
            [](const HeadsUpState& state) {
                py::set values;
                for (int player = 0; player < PLAYERS; ++player) {
                    if (state.pending_mask & (1 << player)) values.add(player);
                }
                return values;
            }
        )
        .def_property_readonly("history", &history_vector)
        .def_readwrite("terminal", &HeadsUpState::terminal)
        .def_property_readonly(
            "payoffs",
            [](const HeadsUpState& state) -> py::object {
                return state.has_payoffs
                    ? py::cast(array_vector(state.payoffs))
                    : py::none();
            }
        )
        .def_property_readonly(
            "payouts",
            [](const HeadsUpState& state) -> py::object {
                return state.has_payouts
                    ? py::cast(array_vector(state.payouts))
                    : py::none();
            }
        )
        .def_property_readonly(
            "winners",
            [](const HeadsUpState& state) {
                return py::tuple(py::cast(state.winners.vector()));
            }
        )
        .def_property_readonly(
            "players_remaining",
            &HeadsUpState::players_remaining
        )
        .def_property_readonly(
            "contrib",
            [](const HeadsUpState& state) {
                return array_vector(state.street_contrib);
            }
        );

    py::class_<HeadsUpHoldemEngine>(module, "HeadsUpHoldemEngine")
        .def(
            py::init<Chip, Chip, Chip, py::object>(),
            py::arg("starting_stack") = 200,
            py::arg("small_blind") = 1,
            py::arg("big_blind") = 2,
            py::arg("seed") = py::none()
        )
        .def_readonly("starting_stack", &HeadsUpHoldemEngine::starting_stack)
        .def_readonly("small_blind", &HeadsUpHoldemEngine::small_blind)
        .def_readonly("big_blind", &HeadsUpHoldemEngine::big_blind)
        .def_readonly("stack_size", &HeadsUpHoldemEngine::stack_size)
        .def_readonly("sb", &HeadsUpHoldemEngine::sb)
        .def_readonly("bb", &HeadsUpHoldemEngine::bb)
        .def_readwrite("rng", &HeadsUpHoldemEngine::rng)
        .def_property(
            "_last_button",
            [](const HeadsUpHoldemEngine& engine) {
                return engine.last_button;
            },
            [](HeadsUpHoldemEngine& engine, int value) {
                engine.last_button = value;
            }
        )
        .def(
            "new_hand",
            &HeadsUpHoldemEngine::new_hand,
            py::arg("button") = py::none(),
            py::kw_only(),
            py::arg("stacks") = py::none(),
            py::arg("deck") = py::none()
        )
        .def_static("clone", &HeadsUpHoldemEngine::clone)
        .def(
            "amount_to_call",
            &HeadsUpHoldemEngine::amount_to_call,
            py::arg("state"),
            py::arg("player") = -1
        )
        .def("legal_actions", &HeadsUpHoldemEngine::legal_actions)
        .def(
            "legal_action_mask",
            &HeadsUpHoldemEngine::legal_action_mask
        )
        .def("action_target", &HeadsUpHoldemEngine::action_target)
        .def("action_payment", &HeadsUpHoldemEngine::action_payment)
        .def("action_descriptors", &HeadsUpHoldemEngine::action_descriptors)
        .def("step", &HeadsUpHoldemEngine::step)
        .def(
            "step_batch",
            [](HeadsUpHoldemEngine& engine,
               const std::vector<HeadsUpState>& states,
               const std::vector<int>& actions) {
                if (states.size() != actions.size()) {
                    throw py::value_error(
                        "states and actions must have identical lengths"
                    );
                }
                std::vector<HeadsUpState> results;
                results.reserve(states.size());
                py::gil_scoped_release release;
                for (std::size_t index = 0; index < states.size(); ++index) {
                    results.push_back(engine.step(states[index], actions[index]));
                }
                return results;
            },
            py::arg("states"),
            py::arg("actions")
        )
        .def(
            "step_exact",
            &HeadsUpHoldemEngine::step_exact,
            py::arg("state"),
            py::arg("kind"),
            py::arg("raise_to") = py::none()
        )
        .def("resolve_showdown", &HeadsUpHoldemEngine::resolve_showdown)
        .def("terminal_payoff", &HeadsUpHoldemEngine::terminal_payoff);

    module.attr("NUM_PLAYERS") = PLAYERS;
    module.attr("NUM_ACTIONS") = ACTIONS;
    module.attr("ENGINE_SCHEMA_VERSION") = "hu_nlhe_engine_v1";
    module.attr("ACTION_SCHEMA_VERSION") = "hu_nlhe_actions_v1_10";
    module.attr("ENCODER_SCHEMA_VERSION") = "hu_information_state_v1";
    module.attr("NATIVE_ABI_VERSION") = 4;
    module.attr("DEFAULT_MAX_HISTORY") = 32;
    module.attr("ACTION_NAMES") = py::cast(ACTION_NAMES);
    module.attr("ACTION_FOLD") = FOLD;
    module.attr("ACTION_CHECK") = CHECK;
    module.attr("ACTION_CALL") = CALL;
    module.attr("ACTION_MIN_RAISE") = MIN_RAISE;
    module.attr("ACTION_THIRD_POT") = THIRD_POT;
    module.attr("ACTION_HALF_POT") = HALF_POT;
    module.attr("ACTION_THREE_QUARTER_POT") = THREE_QUARTER_POT;
    module.attr("ACTION_POT") = POT;
    module.attr("ACTION_OVERBET") = OVERBET;
    module.attr("ACTION_ALL_IN") = ALL_IN;
    module.attr("STREET_PREFLOP") = PREFLOP;
    module.attr("STREET_FLOP") = FLOP;
    module.attr("STREET_TURN") = TURN;
    module.attr("STREET_RIVER") = RIVER;
    module.def("evaluate_5card", &evaluate_5card);
    module.def("evaluate_7card", &evaluate_7card);
    module.def(
        "bayesian_condition",
        &bayesian_condition,
        py::arg("weights"),
        py::arg("likelihoods"),
        py::arg("likelihood_floor") = 1e-6
    );
    module.def(
        "regret_match_root",
        &regret_match_root,
        py::arg("regrets"),
        py::arg("allowed"),
        py::arg("value_scores")
    );
    module.def(
        "hierarchical_regret_match_root",
        &hierarchical_regret_match_root,
        py::arg("regrets"),
        py::arg("allowed"),
        py::arg("value_scores"),
        py::arg("families")
    );
    module.def(
        "estimate_terminal_call_scenarios",
        &estimate_terminal_call_scenarios,
        py::arg("hero_hole"),
        py::arg("board"),
        py::arg("opponent_holes"),
        py::arg("weights"),
        py::arg("fold_payoff"),
        py::arg("win_payoff"),
        py::arg("tie_payoff"),
        py::arg("loss_payoff"),
        py::arg("nominal_samples") = 50000,
        py::arg("seed") = 0
    );
    module.def(
        "estimate_all_in_ev",
        &estimate_all_in_ev,
        py::arg("hero_hole"),
        py::arg("board"),
        py::arg("opponent_holes"),
        py::arg("weights"),
        py::arg("call_probabilities"),
        py::arg("fold_payoff"),
        py::arg("win_payoff"),
        py::arg("tie_payoff"),
        py::arg("loss_payoff"),
        py::arg("samples") = 50000,
        py::arg("seed") = 0,
        py::arg("robust_best_response") = false
    );
    module.def("state_to_dict", &state_to_dict);
    module.def("state_from_dict", &state_from_dict);
    module.def(
        "encode_information_state",
        &encode_information_state_native,
        py::arg("state"),
        py::arg("hero"),
        py::arg("legal_actions"),
        py::arg("big_blind"),
        py::arg("max_history") = 32,
        py::kw_only(),
        py::arg("action_descriptors") = py::none()
    );
    module.attr("HeadsUpHoldemEnv") = module.attr("HeadsUpHoldemEngine");
    module.attr("HeadsUpPokerEnv") = module.attr("HeadsUpHoldemEngine");
    module.attr("GameState") = module.attr("HeadsUpState");
}
