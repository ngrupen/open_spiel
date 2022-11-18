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

inline constexpr int kSearchDepth = 100;
// Counts the number of states in the game according to various measures.
//   - histories is a sequence of moves (for all players) and chance outcomes
//   - states is for imperfect information games, information states (i.e.
//     sets of histories which are indistinguishable to the acting player);
//     for example in poker, the acting player's private cards plus the sequence
//     of bets and public cards, for perfect information games, Markov states
//     (i.e. sets of histories which yield the same result with the same actions
//     applied), e.g. in tic-tac-toe the current state of the board, regardless
//     of the order in which the moves were played.

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(std::string, save_path, "", "Path to save test results.");


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


    // empty function for minimax oracle
    std::function<double(const State&)> empty_funct;

    std::map<std::string, std::unique_ptr<open_spiel::State>> all_state_ids =
        GetAllStateIDs(*game, /*depth_limit=*/-1, /*include_terminals=*/true,
                        /*include_chance_states=*/true, /*stop_at_duplicates=*/false);

    std::vector<std::string> nonterminal_state_strings;
    std::vector<std::string> terminal_state_strings;
    //std::vector<std::vector<double>> all_oracle_values;
    std::vector<double> all_oracle_values;
    std::vector<Action> all_oracle_moves;
    // std::vector<std::vector<double>> terminal_values;
    for (auto const & [state_id_str, state] : all_state_ids) {
      switch (state->GetType()) {
        case StateType::kDecision: {
          if (game->GetType().information == GameType::Information::kPerfectInformation) {
            Player player = state->CurrentPlayer();

            // oracle evaluation
            std::pair<double, Action> value_action = SupergameAlphaBetaSearch(
                *game, state.get(), empty_funct, kSearchDepth, player);
            // std::vector<double> oracle_values;
            // if (player == 0) {
            //     oracle_values.push_back(value_action.first);
            //     oracle_values.push_back(-value_action.first);
            // } else {
            //     oracle_values.push_back(-value_action.first);
            //     oracle_values.push_back(value_action.first);
            // }
            // all_oracle_values.push_back(oracle_values);
            all_oracle_values.push_back(value_action.first);

            all_oracle_moves.push_back(value_action.second);

            // store results
            nonterminal_state_strings.push_back(state_id_str);
          }
          break;
        }
        case StateType::kTerminal: {
          // evaluate terminal state
          // NOTE: can't evaluate terminal state
        //   std::cout << state->CurrentPlayer() << std::endl;
        //   std::vector<double> az_values = az_evaluator->Evaluate(*state);

          // store results
          terminal_state_strings.push_back(state_id_str);
          break;
        }
        case StateType::kMeanField:
          open_spiel::SpielFatalError("kMeanField not handeled.");
      }
    }
    const int num_nonterminal_states = nonterminal_state_strings.size();
    const int num_terminal_states = terminal_state_strings.size();
    std::cout << "Game: " << game_name
              << "\n\t-num_nonterminal_states: " << num_nonterminal_states
              << "\n\t-num_terminal_states: " << num_terminal_states
              << std::endl;

    // construct output file
    std::string save_path = absl::GetFlag(FLAGS_save_path);
    save_path += "/state_data_labels.txt";
    std::cout << "save path: " << save_path << std::endl;

    // save histories and values to file
    std::ofstream outFile;
    outFile.open(save_path);
    if (outFile.is_open()) {
      outFile << "State ID, Empty, Cross, Nought, Value, Move" << std::endl;
      for (int idx = 0; idx < nonterminal_state_strings.size(); ++idx) {
        std::string state_str = nonterminal_state_strings[idx];
        std::string empty = "";
        std::string cross = "";
        std::string nought = "";
        for (int i=0; i<9; i++) {
          if (state_str[i]=='.') {
            empty += "1,";
            cross += "0,";
            nought += "0,";
          } 
          else if (state_str[i]=='x') {
            empty += "0,";
            cross += "1,";
            nought += "0,";
          } 
          else if (state_str[i]=='o') {
            empty += "0,";
            cross += "0,";
            nought += "1,";
          } 
          else open_spiel::SpielFatalError("Unexpected character in state string.");
        }
        empty.pop_back();
        cross.pop_back();
        nought.pop_back();

        outFile << nonterminal_state_strings[idx] << ", " 
                << empty << ", " << cross << ", " << nought << ", " 
                << all_oracle_values[idx] << ", " 
                << all_oracle_moves[idx] 
                << std::endl;
      }
    }
    outFile.close();
}