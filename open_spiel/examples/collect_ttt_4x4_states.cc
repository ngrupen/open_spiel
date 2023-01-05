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
ABSL_FLAG(std::string, player, "random", "Who controls player.");
ABSL_FLAG(std::string, az_path, "", "Path to AZ experiment.");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb",
          "AZ graph definition file name.");
ABSL_FLAG(double, uct_c, 1, "UCT exploration constant.");
ABSL_FLAG(int, rollout_count, 10, "How many rollouts per evaluation.");
ABSL_FLAG(int, max_simulations, 0, "How many simulations to run for AZ.");
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
ABSL_FLAG(int, move_log_cutoff, 7, "Number of moves before logging states.");

uint_fast32_t Seed() {
  uint_fast32_t seed = absl::GetFlag(FLAGS_seed);
  return seed != 0 ? seed : absl::ToUnixMicros(absl::Now());
}

std::unique_ptr<open_spiel::Bot>
InitAZBot(std::string type, const open_spiel::Game &game,
        open_spiel::Player player,
        std::shared_ptr<open_spiel::algorithms::torch_az::VPNetEvaluator>
            az_evaluator) {
  if (type == "az") {
    return std::make_unique<open_spiel::algorithms::MCTSBot>(
        game, std::move(az_evaluator), absl::GetFlag(FLAGS_uct_c),
        absl::GetFlag(FLAGS_max_simulations),
        absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
        absl::GetFlag(FLAGS_verbose));
  }
  open_spiel::SpielFatalError(
      "Bad player type. Known types: az, human, mcts, random");
}

std::unique_ptr<open_spiel::Bot>
InitBot(std::string type, const open_spiel::Game &game,
        open_spiel::Player player,
        std::shared_ptr<open_spiel::algorithms::Evaluator> evaluator) {
  if (type == "human") {
    return std::make_unique<open_spiel::HumanBot>();
  }
  if (type == "mcts") {
    return std::make_unique<open_spiel::algorithms::MCTSBot>(
        game, std::move(evaluator), absl::GetFlag(FLAGS_uct_c),
        absl::GetFlag(FLAGS_max_simulations),
        absl::GetFlag(FLAGS_max_memory_mb), absl::GetFlag(FLAGS_solve), Seed(),
        absl::GetFlag(FLAGS_verbose));
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


std::tuple<std::vector<double>, std::vector<std::string>, std::vector<std::string>>
PlayGame(const open_spiel::Game &game,
         std::vector<std::unique_ptr<open_spiel::Bot>> &bots, std::mt19937 &rng) {
  bool quiet = absl::GetFlag(FLAGS_quiet);
  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<std::string> history;
  std::vector<std::string> state_history;
  int num_moves = 0;

  if (!quiet)
    std::cerr << "Initial state:\n" << state << std::endl;

  // play random moves (if any)
  int random_moves = absl::GetFlag(FLAGS_num_random_moves);
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

  int move_log_cutoff = absl::GetFlag(FLAGS_move_log_cutoff);
  while (!state->IsTerminal()) {
    open_spiel::Player current_player = state->CurrentPlayer();
    open_spiel::Player opponent_player = 1 - current_player;

    // The state must be a decision node, ask the right bot to make its action.
    open_spiel::Action action = bots[current_player]->Step(*state);

    if (!quiet)
      std::cerr << "Player " << current_player << " chose action: "
                << state->ActionToString(current_player, action) << std::endl;

    // Inform the other bot of the action performed.
    bots[opponent_player]->InformAction(*state, current_player, action);

    // Update history and get the next state.
    history.push_back(state->ActionToString(current_player, action));
    state->ApplyAction(action);
    if ((num_moves > move_log_cutoff) && (!state->IsTerminal())) {
        state_history.push_back(state->GetIDString());
    }
    num_moves++;

    if (!quiet)
      std::cerr << "Next state:\n" << state->ToString() << std::endl;
  }

  std::cerr << "Returns: " << absl::StrJoin(state->Returns(), ", ")
            << std::endl;
  std::cerr << "Game actions: " << absl::StrJoin(history, ", ") << std::endl;

  return {state->Returns(), history, state_history};
}

int main(int argc, char **argv) {
  std::vector<char *> positional_args = absl::ParseCommandLine(argc, argv);
  std::mt19937 rng(Seed());  // Random number generator.

  // Create the game.
  std::string game_name = absl::GetFlag(FLAGS_game);
  std::cerr << "Game: " << game_name << std::endl;
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(game_name);

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

  std::vector<std::unique_ptr<open_spiel::Bot>> bots;
  open_spiel::algorithms::torch_az::DeviceManager device_manager;
  if (absl::GetFlag(FLAGS_player) == "az") {
    if (absl::GetFlag(FLAGS_az_path).empty())
        open_spiel::SpielFatalError("AlphaZero path must be specified for AZ player.");

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
    bots.push_back(
        InitAZBot(absl::GetFlag(FLAGS_player), *game, 0, az_evaluator));
    bots.push_back(
        InitAZBot(absl::GetFlag(FLAGS_player), *game, 1, az_evaluator));
  } else {
    auto evaluator =
      std::make_shared<open_spiel::algorithms::RandomRolloutEvaluator>(
          absl::GetFlag(FLAGS_rollout_count), Seed());
    
    bots.push_back(
        InitBot(absl::GetFlag(FLAGS_player), *game, 0, evaluator));
    bots.push_back(
        InitBot(absl::GetFlag(FLAGS_player), *game, 1, evaluator));
  }

  // game params set up
  std::map<std::string, int> histories;
  std::vector<double> overall_returns(2, 0);
  std::vector<int> overall_wins(2, 0);
  int num_games = absl::GetFlag(FLAGS_num_games);

  // construct output files
  std::string save_path = absl::GetFlag(FLAGS_save_path);
  save_path += "/states.txt";

  std::ofstream outFile;
  outFile.open(save_path);
  if (outFile.is_open()) {
    for (int game_num = 0; game_num < num_games; ++game_num) {
      // play game
      auto [returns, history, state_history] = PlayGame(*game, bots, rng);

      // log info
      std::cerr << "Game " << game_num << std::endl;
      std::cerr << "History: " << absl::StrJoin(history, ",") << std::endl;
    
      for (int idx = 0; idx < state_history.size(); ++idx) {
        outFile << state_history[idx] << std::endl;
      }

      // log history
      histories[absl::StrJoin(history, " ")] += 1;
      for (int i = 0; i < returns.size(); ++i) {
        double v = returns[i];
        overall_returns[i] += v;
        if (v > 0) {
          overall_wins[i] += 1;
        }
      }
    }
  } else {
    open_spiel::SpielFatalError("Couldn't open out file!");
  }

  std::cerr << "Number of games played: " << num_games << std::endl;
  std::cerr << "Number of distinct games played: " << histories.size()  << std::endl;
  std::cerr << "Players: " << absl::GetFlag(FLAGS_player) << ", " << absl::GetFlag(FLAGS_player) << std::endl;
  std::cerr << "Overall wins: " << absl::StrJoin(overall_wins, ", ") << std::endl;
  std::cerr << "Overall returns: " << absl::StrJoin(overall_returns, ", ") << std::endl;

  return 0;
<<<<<<< HEAD
}
=======
}
>>>>>>> f3c28fda7c5605da9206989eb7191f8bc014f8d9
