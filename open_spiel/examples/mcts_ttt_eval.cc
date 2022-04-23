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

#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/human/human_bot.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

ABSL_FLAG(std::string, game, "tic_tac_toe", "The name of the game to play.");
ABSL_FLAG(std::string, player1, "mcts", "Who controls player1.");
ABSL_FLAG(std::string, player2, "mcts", "Who controls player2.");
ABSL_FLAG(std::string, az_path, "", "Path to AZ experiment.");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb",
          "AZ graph definition file name.");
ABSL_FLAG(double, uct_c, 2, "UCT exploration constant.");
ABSL_FLAG(int, rollout_count, 10, "How many rollouts per evaluation.");
ABSL_FLAG(int, max_az_simulations, 2, "How many simulations to run for AZ.");
ABSL_FLAG(int, max_op_simulations, 20000, "How many simulations to run for opponent.");
ABSL_FLAG(int, num_games, 1, "How many games to play.");
ABSL_FLAG(int, num_random_moves, 0, "Number of random starting moves.");
ABSL_FLAG(int, max_memory_mb, 1000,
          "The maximum memory used before cutting the search short.");
ABSL_FLAG(int, az_checkpoint, -1, "Checkpoint of AZ model.");
ABSL_FLAG(int, az_batch_size, 1, "Batch size of AZ inference.");
ABSL_FLAG(int, az_threads, 1, "Number of threads to run for AZ inference.");
ABSL_FLAG(int, az_cache_size, 16384, "Cache size of AZ algorithm.");
ABSL_FLAG(int, az_cache_shards, 1, "Cache shards of AZ algorithm.");
ABSL_FLAG(bool, solve, true, "Whether to use MCTS-Solver.");
ABSL_FLAG(uint_fast32_t, seed, 0, "Seed for MCTS.");
ABSL_FLAG(bool, verbose, false, "Show the MCTS stats of possible moves.");
ABSL_FLAG(bool, quiet, false, "Show the MCTS stats of possible moves.");
ABSL_FLAG(std::string, save_path, "", "Path to save test results.");


uint_fast32_t Seed() {
  uint_fast32_t seed = absl::GetFlag(FLAGS_seed);
  return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

std::unique_ptr<open_spiel::Bot>
InitBot(std::string type, const open_spiel::Game &game,
        open_spiel::Player player,
        std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator,
        std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator>
            az_evaluator,
        bool player1) {
  if (type == "az") {
    return std::make_unique<open_spiel::algorithms::MCTSBot>(
        game, std::move(az_evaluator), absl::GetFlag(FLAGS_uct_c),
        absl::GetFlag(FLAGS_max_az_simulations),
        absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
        absl::GetFlag(FLAGS_verbose));
  }
  if (type == "human") {
    return std::make_unique<open_spiel::HumanBot>();
  }
  if (type == "mcts") {
    if(player1) {
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(evaluator), absl::GetFlag(FLAGS_uct_c),
            absl::GetFlag(FLAGS_max_az_simulations),
            absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
            absl::GetFlag(FLAGS_verbose));
    }
    else {
        return std::make_unique<open_spiel::algorithms::MCTSBot>(
            game, std::move(evaluator), absl::GetFlag(FLAGS_uct_c),
            absl::GetFlag(FLAGS_max_op_simulations),
            absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
            absl::GetFlag(FLAGS_verbose));
    }
  }
  if (type == "random") {
    return open_spiel::MakeUniformRandomBot(player, Seed());
  }

  open_spiel::SpielFatalError(
      "Bad player type. Known types: az, human, mcts, random");
}

open_spiel::Action GetAction(const open_spiel::State &state,
                             std::string action_str) {
  for (open_spiel::Action action : state.LegalActions()) {
    if (action_str == state.ActionToString(state.CurrentPlayer(), action))
      return action;
  }
  return open_spiel::kInvalidAction;
}


std::tuple<std::vector<double>, std::vector<std::string>, std::vector<open_spiel::Action>, std::vector<std::string>, std::vector<int>>
PlayGame(const open_spiel::Game &game,
         std::vector<std::unique_ptr<open_spiel::Bot>> &bots, std::mt19937 &rng,
         const std::vector<std::string> &initial_actions, int az_player) {
  bool quiet = absl::GetFlag(FLAGS_quiet);
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<std::string> history;

  std::vector<open_spiel::Action> az_actions;
  std::vector<std::string> az_action_strs;
  std::vector<int> az_action_scores;

  if (!quiet)
    std::cerr << "Initial state:\n" << state << std::endl;

  // play the initial actions (if there are any).
  for (const auto &action_str : initial_actions) {
    open_spiel::Player current_player = state->CurrentPlayer();
    open_spiel::Action action = GetAction(*state, action_str);

    if (action == open_spiel::kInvalidAction)
      open_spiel::SpielFatalError(absl::StrCat("Invalid action: ", action_str));

    history.push_back(action_str);
    state->ApplyAction(action);

    if (!quiet) {
      std::cerr << "Player " << current_player
                << " forced action: " << action_str << std::endl;
      std::cerr << "Next state:\n" << state->ToString() << std::endl;
    }
  }

  // play random moves (if any) 
  int random_moves = rand() % 5;
  std::random_device rd;
  std::default_random_engine eng(rd());
  if (random_moves > 0) {
    for (int i=0; i < random_moves; ++i) {
      open_spiel::Player current_player = state->CurrentPlayer();
      std::vector<open_spiel::Action> real_legal_moves = state->OriginalLegalActions(current_player);
      std::uniform_int_distribution<int> distr(0, real_legal_moves.size()-1);
      int idx = distr(eng);
      open_spiel::Action rand_action = real_legal_moves[idx];

      std::string temp_action_str = state->ActionToString(current_player, rand_action);

      if (rand_action == open_spiel::kInvalidAction)
        open_spiel::SpielFatalError(absl::StrCat("Invalid action: ", temp_action_str));

      history.push_back(temp_action_str);
      state->ApplyAction(rand_action);

      if (!quiet) {
        std::cerr << "Player " << current_player
                  << " forced action: " << temp_action_str << std::endl;
        std::cerr << "Next state:\n" << state->ToString() << std::endl;
      }
    }
  }

  open_spiel::GameType game_type = game.GetType();

  while (!state->IsTerminal()) {
    open_spiel::Player current_player = state->CurrentPlayer();
    open_spiel::Player opponent_player = 1 - current_player;

    std::cerr << "AZ Player: " << az_player << std::endl;
    std::cerr << "Current Player: " << current_player << std::endl;

    // The state must be a decision node, ask the right bot to make its action.
    open_spiel::Action action = bots[current_player]->Step(*state);

    // save az_player's actions
    if (az_player == current_player) {
      std::cerr << "Action: " << action << std::endl;
      az_actions.push_back(action);
      az_action_strs.push_back(state->ActionToString(current_player, action));
      int valid;
      if (game_type.short_name == "tic_tac_toe")
        valid = 0;
      else
        valid = state->CheckValidMove(current_player, action);
      std::cerr << "Valid move: " << valid << std::endl;
      az_action_scores.push_back(valid);
    }

    if (!quiet)
      std::cerr << "Player " << current_player << " chose action: "
                << state->ActionToString(current_player, action) << std::endl;

    // Inform the other bot of the action performed.
    bots[opponent_player]->InformAction(*state, current_player, action);

    // Update history and get the next state.
    history.push_back(state->ActionToString(current_player, action));
    state->ApplyAction(action);

    if (!quiet)
      std::cerr << "Next state:\n" << state->ToString() << std::endl;
  }

  std::cerr << "Returns: " << absl::StrJoin(state->Returns(), ", ")
            << std::endl;
  std::cerr << "Game actions: " << absl::StrJoin(history, ", ") << std::endl;

  return {state->Returns(), history, az_actions, az_action_strs, az_action_scores};
}

std::vector<std::string> tokenize(std::string const &str, const char delim) {
    size_t start;
    size_t end = 0;
    std::vector<std::string> out;
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
    return out;
}

bool isNumber(const std::string& str)
{
    return str.find_first_not_of("0123456789") == std::string::npos;
}

int main(int argc, char **argv) {
  std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);
  std::mt19937 rng(Seed());  // Random number generator.

  // Create the game.
  std::string game_name = absl::GetFlag(FLAGS_game);
  std::cerr << "Game: " << game_name << std::endl;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_name);
  open_spiel::GameType game_type = game->GetType();

  // Ensure the game is AlphaZero-compatible and arguments are compatible.
  if (game->NumPlayers() != 2)
    open_spiel::SpielFatalError("AlphaZero can only handle 2-player games.");
  if (game_type.reward_model != open_spiel::GameType::RewardModel::kTerminal)
    open_spiel::SpielFatalError("Game must have terminal rewards.");
  if (game_type.dynamics != open_spiel::GameType::Dynamics::kSequential)
    open_spiel::SpielFatalError("Game must have sequential turns.");
  if (game_type.chance_mode != open_spiel::GameType::ChanceMode::kDeterministic)
    open_spiel::SpielFatalError("Game must be deterministic.");
//   if (absl::GetFlag(FLAGS_az_path).empty())
//     open_spiel::SpielFatalError("AlphaZero path must be specified.");
//   if (absl::GetFlag(FLAGS_player1) != "az" &&
//       absl::GetFlag(FLAGS_player2) != "az")
//     open_spiel::SpielFatalError("One of the players must be AlphaZero.");

    
    std::cout << "playing games with checkpoint: " << absl::GetFlag(FLAGS_az_checkpoint) << std::endl;

    open_spiel::algorithms::torch_az::DeviceManager device_manager;
    device_manager.AddDevice(open_spiel::algorithms::torch_az::VPNetModel(
        *game, absl::GetFlag(FLAGS_az_path), absl::GetFlag(FLAGS_az_graph_def),
        "/cpu:0"));
    device_manager.Get(0, 0)->LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));

    auto az_evaluator =
        std::make_shared<open_spiel::algorithms::torch_az::VPNetEvaluator>(
            /*device_manager=*/&device_manager,
            /*batch_size=*/absl::GetFlag(FLAGS_az_batch_size),
            /*threads=*/absl::GetFlag(FLAGS_az_threads),
            /*cache_size=*/absl::GetFlag(FLAGS_az_cache_size),
            /*cache_shards=*/absl::GetFlag(FLAGS_az_cache_shards));
    auto evaluator =
        std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
            absl::GetFlag(FLAGS_rollout_count), Seed());

    std::vector<std::unique_ptr<open_spiel::Bot>> bots;
    bots.push_back(
        InitBot(absl::GetFlag(FLAGS_player1), *game, 0, evaluator, az_evaluator, true));
    bots.push_back(
        InitBot(absl::GetFlag(FLAGS_player2), *game, 1, evaluator, az_evaluator, false));

    std::vector<std::string> initial_actions;
    for (int i = 1; i < positional_args.size(); ++i) {
        initial_actions.push_back(positional_args[i]);
    }

    int az_player;

    // if (absl::GetFlag(FLAGS_player1) == "az") {
        az_player = 0;
    // }
    // else {
        // az_player = 1;
    // }

    std::map<std::string, int> histories;
    std::vector<double> overall_returns(2, 0);
    std::vector<int> overall_wins(2, 0);
    int num_games = absl::GetFlag(FLAGS_num_games);

    // construct output file
    std::string save_path = absl::GetFlag(FLAGS_save_path);
    save_path += "/checkpoint_";
    save_path += std::to_string(absl::GetFlag(FLAGS_az_checkpoint));
    save_path += ".txt";

    std::ofstream outFile;
    outFile.open(save_path);
    if (outFile.is_open()) {
        for (int game_num = 0; game_num < num_games; ++game_num) {
        auto [returns, history, az_actions, az_action_strs, az_action_scores] = PlayGame(*game, bots, rng, initial_actions, az_player);
        std::cerr << "Game " << game_num << std::endl;
        std::cerr << "History: " << absl::StrJoin(history, ",") << std::endl;
        std::cerr << "AZ actions: " << absl::StrJoin(az_actions, ", ") << std::endl;
        std::cerr << "AZ action strings: " << absl::StrJoin(az_action_strs, ", ") << std::endl;
        std::cerr << "AZ action scores: " << absl::StrJoin(az_action_scores, ", ") << std::endl;

        outFile << "History: " << absl::StrJoin(history, ",") << std::endl;
        outFile << "AZ actions: " << absl::StrJoin(az_actions, ",") << std::endl;
        outFile << "AZ action strings: " << absl::StrJoin(az_action_strs, ",") << std::endl;
        outFile << "AZ action scores: " << absl::StrJoin(az_action_scores, ",") << std::endl;
        histories[absl::StrJoin(history, " ")] += 1;
        for (int i = 0; i < returns.size(); ++i) {
            double v = returns[i];
            overall_returns[i] += v;
            if (v > 0) {
            overall_wins[i] += 1;
            }
        }
        }
    }

    std::cerr << "Number of games played: " << num_games << std::endl;
    std::cerr << "Number of distinct games played: " << histories.size()  << std::endl;
    std::cerr << "Players: " << absl::GetFlag(FLAGS_player1) << ", " << absl::GetFlag(FLAGS_player2) << std::endl;
    std::cerr << "Overall wins: " << absl::StrJoin(overall_wins, ", ") << std::endl;
    std::cerr << "Overall returns: " << absl::StrJoin(overall_returns, ", ") << std::endl;

  return 0;
}