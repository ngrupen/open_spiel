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

#include "open_spiel/games/tic_tac_toe_supergame_general.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace tic_tac_toe_supergame_general {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"tic_tac_toe_supergame_general",
    /*long_name=*/"Tic Tac Toe (SuperGame General)",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"num_illegal_actions", 
      GameParameter((GameParameter::Type::kInt, /*is_mandatory=*/true)}},
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TicTacToeSuperGameGeneralGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kCross;
    case 1:
      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kNought:
      return "o";
    case CellState::kCross:
      return "x";
    default:
      SpielFatalError("Unknown state.");
  }
}

void TicTacToeSuperGameGeneralState::DoApplyAction(Action move) {
//   SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
  if (move < kNumCells) {
    board_[move] = PlayerToState(CurrentPlayer());
    if (HasLine(current_player_)) {
      outcome_ = current_player_;
    }
  } else {
    invalid_mover_ = current_player_;
    outcome_ = 1 - current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> TicTacToeSuperGameGeneralState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      moves.push_back(cell);
    }
  }
  for (int cell = kNumCells; cell < (kNumCells + num_illegal_actions_); ++cell) {
    moves.push_back(cell);
  }
  return moves;
}

std::vector<Action> TicTacToeSuperGameGeneralState::OriginalLegalActions(Player player) {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      moves.push_back(cell);
    }
  }
  return moves;
}

int TicTacToeSuperGameGeneralState::CheckValidMove(Player player, Action action) {
  // 0 = valid, 1 = invalid --> can only be invalid if not in the range of illegal actions
  int score = 1;

  if (action < kNumCells)
    score = 0;

  return score;
}

std::string TicTacToeSuperGameGeneralState::ActionToString(Player player,
                                           Action action_id) const {
  if(action_id < kNumCells)
    return absl::StrCat(StateToString(PlayerToState(player)), "(",
                        action_id / kNumCols, ",", action_id % kNumCols, ")");
  else
    return absl::StrCat(StateToString(PlayerToState(player)), "illegal action #", action_id);
}

bool TicTacToeSuperGameGeneralState::HasLine(Player player) const {
  CellState c = PlayerToState(player);
  return (board_[0] == c && board_[1] == c && board_[2] == c) ||
         (board_[3] == c && board_[4] == c && board_[5] == c) ||
         (board_[6] == c && board_[7] == c && board_[8] == c) ||
         (board_[0] == c && board_[3] == c && board_[6] == c) ||
         (board_[1] == c && board_[4] == c && board_[7] == c) ||
         (board_[2] == c && board_[5] == c && board_[8] == c) ||
         (board_[0] == c && board_[4] == c && board_[8] == c) ||
         (board_[2] == c && board_[4] == c && board_[6] == c);
}

bool TicTacToeSuperGameGeneralState::IsFull() const { return num_moves_ == kNumCells; }

TicTacToeSuperGameGeneralState::TicTacToeSuperGameGeneralState(std::shared_ptr<const Game> game, int num_illegal_actions) 
    : State(game),
      num_illegal_actions_(num_illegal_actions) {
    std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string TicTacToeSuperGameGeneralState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool TicTacToeSuperGameGeneralState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> TicTacToeSuperGameGeneralState::Returns() const {
  if ((HasLine(Player{0}) && invalid_mover_ != Player{0}) || (invalid_mover_ == Player{1})) {
    return {1.0, -1.0};
  } else if ((HasLine(Player{1}) && invalid_mover_ != Player{1}) || (invalid_mover_ == Player{0})) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

// std::vector<double> TicTacToeNoRulesState::Returns() const {
//   if (HasLine(Player{0})) {
//     return {1.0, -1.0};
//   } else if (HasLine(Player{1})) {
//     return {-1.0, 1.0};
//   } else {
//     return {0.0, 0.0};
//   }
// }

std::string TicTacToeSuperGameGeneralState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TicTacToeSuperGameGeneralState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TicTacToeSuperGameGeneralState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void TicTacToeSuperGameGeneralState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> TicTacToeSuperGameGeneralState::Clone() const {
  return std::unique_ptr<State>(new TicTacToeSuperGameGeneralState(*this));
}

TicTacToeSuperGameGeneralGame::TicTacToeSuperGameGeneralGame(const GameParameters& params)
    : Game(kGameType, params),
      num_illegal_actions_(ParameterValue<int>("num_illegal_actions")) {}

}  // namespace tic_tac_toe_supergame_general
}  // namespace open_spiel
