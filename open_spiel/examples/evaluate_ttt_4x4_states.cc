// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <memory>
#include <fstream>
#include <iterator>
#include <sstream>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_set.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/get_all_histories.h"
#include "open_spiel/algorithms/get_all_state_ids.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/algorithms/supergame_minimax.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

using open_spiel::LoadGame;
using open_spiel::GameType;
using open_spiel::StateType;
using open_spiel::Player;
using open_spiel::State;
using open_spiel::Action;
using open_spiel::algorithms::GetAllHistories;
using open_spiel::algorithms::GetAllStateIDs;
using open_spiel::algorithms::SupergameAlphaBetaSearch;

inline constexpr int kSearchDepth = 1000;
// Counts the number of states in the game according to various measures.
//   - histories is a sequence of moves (for all players) and chance outcomes
//   - states is for imperfect information games, information states (i.e.
//     sets of histories which are indistinguishable to the acting player);
//     for example in poker, the acting player's private cards plus the sequence
//     of bets and public cards, for perfect information games, Markov states
//     (i.e. sets of histories which yield the same result with the same actions
//     applied), e.g. in tic-tac-toe the current state of the board, regardless
//     of the order in which the moves were played.

ABSL_FLAG(std::string, game, "tic_tac_toe_4x4", "The name of the game to play.");
ABSL_FLAG(std::string, az_path, "", "Path to AZ experiment.");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb", "AZ graph definition file name.");
ABSL_FLAG(int, az_checkpoint, -1, "Checkpoint of AZ model.");
ABSL_FLAG(int, az_batch_size, 1, "Batch size of AZ inference.");
ABSL_FLAG(int, az_threads, 1, "Number of threads to run for AZ inference.");
ABSL_FLAG(int, az_cache_size, 16384, "Cache size of AZ algorithm.");
ABSL_FLAG(int, az_cache_shards, 1, "Cache shards of AZ algorithm.");
ABSL_FLAG(std::string, data_path, "", "Path to 4x4 state ids.");
ABSL_FLAG(std::string, save_path, "", "Path to save test results.");

std::vector<double> Softmax(
    const std::vector<double>& values, double lambda) {
  std::vector<double> new_values = values;
  for (double& new_value : new_values) {
    new_value *= lambda;
  }
  double max = *std::max_element(new_values.begin(), new_values.end());

  double denom = 0;
  for (int idx = 0; idx < values.size(); ++idx) {
    new_values[idx] = std::exp(new_values[idx] - max);
    denom += new_values[idx];
  }

  SPIEL_CHECK_GT(denom, 0);
  double prob_sum = 0.0;
  std::vector<double> policy;
  policy.reserve(new_values.size());
  for (int idx = 0; idx < values.size(); ++idx) {
    double prob = new_values[idx] / denom;
    SPIEL_CHECK_PROB(prob);
    prob_sum += prob;
    policy.push_back(prob);
  }

  SPIEL_CHECK_FLOAT_NEAR(prob_sum, 1.0, 1e-12);
  return policy;
}

int main(int argc, char** argv) {
    std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);

    // load game
    std::string game_name = absl::GetFlag(FLAGS_game);
    std::cerr << "Game: " << game_name << std::endl;
    std::shared_ptr<const open_spiel::Game> game = LoadGame(std::string(game_name));

    // Ensure the game is AlphaZero-compatible and arguments are compatible.
    open_spiel::GameType game_type = game->GetType();
    if (game->NumPlayers() != 2)
      open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
    if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
      open_spiel::SpielFatalError("Game must have terminal rewards.");
    if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
      open_spiel::SpielFatalError("Game must have sequential turns.");
    if (game_type.chance_mode != open_spiel::GameType::ChanceMode::kDeterministic)
      open_spiel::SpielFatalError("Game must be deterministic.");
    std::cout << "Checkpoint path: " << absl::GetFlag(FLAGS_az_path) << std::endl;
    if (absl::GetFlag(FLAGS_az_path).empty())
      open_spiel::SpielFatalError("AlphaZero path must be specified.");

    // init az model
    open_spiel::algorithms::torch_az::DeviceManager device_manager;
    device_manager.AddDevice(open_spiel::algorithms::torch_az::VPNetModel(
      *game, absl::GetFlag(FLAGS_az_path), absl::GetFlag(FLAGS_az_graph_def),
      "/cpu:0"));
    device_manager.Get(0, 0)->LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));
    auto az_evaluator = std::make_shared<open_spiel::algorithms::torch_az::VPNetEvaluator>(
                            /*device_manager=*/&device_manager,
                            /*batch_size=*/absl::GetFlag(FLAGS_az_batch_size),
                            /*threads=*/absl::GetFlag(FLAGS_az_threads),
                            /*cache_size=*/absl::GetFlag(FLAGS_az_cache_size),
                            /*cache_shards=*/absl::GetFlag(FLAGS_az_cache_shards));


    // empty function for minimax oracle
    std::function<double(const State&)> empty_funct;

    std::vector<std::string> state_strs;
    std::ifstream file_in(absl::GetFlag(FLAGS_data_path));

    // Check if object is valid
    if(!file_in)
    {
        open_spiel::SpielFatalError("Cannot open state strings file!");
    }

    std::string line;
    while (std::getline(file_in, line)) {
        if(line.size() > 0) {
            state_strs.push_back(line);
        }
        else {
            open_spiel::SpielFatalError("Empty state string!");
        }
    }

    file_in.close();
    std::cout << "State strs: " << state_strs.size() << std::endl;
    // for (int idx = 0; idx < state_strs.size(); ++idx) {
    //     std::cout << "State ID " << idx << ": " << state_strs[idx] << std::endl;
    //     std::unique_ptr<open_spiel::State> state = game->NewInitialState(state_strs[idx]);
    // }

    // std::map<std::string, std::unique_ptr<open_spiel::State>> all_state_ids =
    //     GetAllStateIDs(*game, /*depth_limit=*/-1, /*include_terminals=*/true,
    //                     /*include_chance_states=*/true, /*stop_at_duplicates=*/false);

    // construct output file
    std::string save_path = absl::GetFlag(FLAGS_save_path);
    save_path += "/state_values.txt";
    std::cout << "save path: " << save_path << std::endl;

    std::vector<std::string> nonterminal_state_strings;
    std::vector<std::string> terminal_state_strings;
    std::vector<std::vector<double>> all_az_values;
    std::vector<std::vector<double>> all_az_policies;
    std::vector<std::vector<double>> all_az_value_policies;
    std::vector<std::vector<Action>> all_legal_moves;
    std::vector<std::vector<double>> all_oracle_values;
    // for (auto const & [state_id_str, state] : all_state_ids) {

    std::ofstream outFile;
    outFile.open(save_path);
    if (outFile.is_open()) {
        for (int idx = 0; idx < state_strs.size(); ++idx) {
            std::cout << "State ID " << idx << ": " << state_strs[idx] << std::endl;
            std::unique_ptr<open_spiel::State> state = game->NewInitialState(state_strs[idx]);

            // current player
            Player player = state->CurrentPlayer();

            // legal actions
            std::vector<Action> legal_moves = state->OriginalLegalActions(player);
            all_legal_moves.push_back(legal_moves);

            // az evaluation of non-terminal state
            std::vector<double> az_values = az_evaluator->Evaluate(*state);
            // all_az_values.push_back(az_values);

            // az policy at non-terminal state
            open_spiel::ActionsAndProbs az_prior = az_evaluator->Prior(*state);
            std::vector<double> az_policy;
            for (auto const & [act, prob] : az_prior) {
                az_policy.push_back(prob);
            }
            // all_az_policies.push_back(az_policy);

            // az value probabilities at non-terminal state
            open_spiel::ActionsAndProbs az_value_policy_probs;
            az_value_policy_probs.reserve(legal_moves.size());
            std::vector<double> next_values;
            for (auto const & act : legal_moves) {
                std::unique_ptr<State> temp_state = state->Clone();
                temp_state->ApplyAction(act);
                
                double temp_value;
                if (temp_state->IsTerminal()) {
                    temp_value = temp_state->Returns().front();
                } else {
                    temp_value = az_evaluator->Evaluate(*temp_state).front();
                }

                next_values.push_back(temp_value);
            }

            std::vector<double> az_value_policy = Softmax(next_values, 1.0);
            // all_az_value_policies.push_back(az_value_policy);

            // oracle evaluation
            std::pair<double, Action> value_action = SupergameAlphaBetaSearch(
                *game, state.get(), empty_funct, kSearchDepth, player);
            std::vector<double> oracle_values;
            if (player == 0) {
                oracle_values.push_back(value_action.first);
                oracle_values.push_back(-value_action.first);
            } else {
                oracle_values.push_back(-value_action.first);
                oracle_values.push_back(value_action.first);
            }
            // all_oracle_values.push_back(oracle_values);

            // store results
            // nonterminal_state_strings.push_back(state_strs[idx]);

            outFile << "State ID: " << state_strs[idx] << std::endl;
            outFile << "AZ Values: " << absl::StrJoin(az_values, ",") << std::endl;
            outFile << "Oracle Values: " << absl::StrJoin(oracle_values, ",") << std::endl;
            outFile << "AZ Policy: " << absl::StrJoin(az_policy, ",") << std::endl;
            outFile << "AZ Value Policy: " << absl::StrJoin(az_value_policy, ",") << std::endl;
            outFile << "Legal Moves: " << absl::StrJoin(legal_moves, ",") << std::endl;
        }
    }
    // construct output file
    // std::string save_path = absl::GetFlag(FLAGS_save_path);
    // save_path += "/state_values.txt";
    // std::cout << "save path: " << save_path << std::endl;

    // save histories and values to file
    // std::ofstream outFile;
    // outFile.open(save_path);
    // if (outFile.is_open()) {
    //   for (int idx = 0; idx < all_az_values.size(); ++idx) {
    //     outFile << "State ID: " << nonterminal_state_strings[idx] << std::endl;
    //     outFile << "AZ Values: " << absl::StrJoin(all_az_values[idx], ",") << std::endl;
    //     outFile << "Oracle Values: " << absl::StrJoin(all_oracle_values[idx], ",") << std::endl;
    //     outFile << "AZ Policy: " << absl::StrJoin(all_az_policies[idx], ",") << std::endl;
    //     outFile << "AZ Value Policy: " << absl::StrJoin(all_az_value_policies[idx], ",") << std::endl;
    //     outFile << "Legal Moves: " << absl::StrJoin(all_legal_moves[idx], ",") << std::endl;
    //   }
    // } else {
    //     open_spiel::SpielFatalError("Couldn't open out file!");
    // }

    outFile.close();
}
