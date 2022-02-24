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

#include "open_spiel/games/tic_tac_toe_supergame_B.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace tic_tac_toe_supergame_B {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"tic_tac_toe_supergame_B",
    /*long_name=*/"Tic Tac Toe (SuperGame B)",
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
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TicTacToeSuperGameBGame(params));
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

// look for both (i) if cell is empty; (ii) if correct marker was played
void TicTacToeSuperGameBState::DoApplyAction(Action move) {
  Action position_id = ActionToPosition(move);
  CellState chosen_player = ActionToMarker(move);

  // std::cerr << "action id: " << move << std::endl;
  // std::cerr << "position id: " << position_id << std::endl;
  // std::cerr << "current player: " << PlayerToState(current_player_) << std::endl;
  // std::cerr << "chosen player: " << chosen_player << std::endl;

  if ((chosen_player == PlayerToState(current_player_)) && (board_[position_id] == CellState::kEmpty)) {
    board_[position_id] = PlayerToState(CurrentPlayer());
    if (HasLine(current_player_)) {
      outcome_ = current_player_;
    }
  } else {
      board_[position_id] = chosen_player;
      invalid_mover_ = current_player_;
      outcome_ = 1 - current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> TicTacToeSuperGameBState::LegalActions() const {
  if (IsTerminal()) return {};

  // can move in any cell, with any marker
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells*2; ++cell) {
    moves.push_back(cell);
  }
  return moves;
}

std::vector<Action> TicTacToeSuperGameBState::OriginalLegalActions(Player player) {
  if (IsTerminal()) return {};

  int offset;
  if (player == 0) {
    offset = 0;
  } else {
    offset = kNumCells;
  }

  // Can move in any empty cell.
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    if (board_[cell] == CellState::kEmpty) {
      moves.push_back(cell + offset);
    }
  }
  return moves;
}

int TicTacToeSuperGameBState::CheckValidMove(Player player, Action action) {
  // 0 = valid, 1 = cell already filled, 2 = wrong marker, 3 = both failures
  int score;

  // check if one of other player's actions was taken
  bool wrong_player;
  if (player == 0) {
    wrong_player = (action >= kNumCells);
  } else {
    wrong_player = (action < kNumCells);
  }
  std::cerr << "player: " << player << ", action: " << action << ", wrong player: " << wrong_player << std::endl;

  // check if player moved on top of existing piece
  bool filled_cell;
  Action pos = ActionToPosition(action);
  if (board_[pos] == CellState::kEmpty) {
      filled_cell = false;
  } else{
      filled_cell = true;
  }
  std::cerr << "player: " << player << ", action: " << action << ", filled cell : " << filled_cell << std::endl;

  if (wrong_player && filled_cell) {
      score = 3;
  } else if (wrong_player && !filled_cell) {
      score = 2;
  } else if (!wrong_player && filled_cell) {
      score = 1;
  } else if (!wrong_player && !filled_cell) {
      score = 0;
  }
  std::cerr << "score: " << score << std::endl;
  return score;
}


CellState ActionToMarker(Action action_id) {
  // std::cerr << "action id: " << action_id << std::endl;

  if (action_id / kNumCells == 0) {
    return CellState::kCross;
  }
  else {
    return CellState::kNought;
  }
}

Action ActionToPosition(Action action_id) {
    return action_id % kNumCells;
}

std::string TicTacToeSuperGameBState::ActionToString(Player player,
                                           Action action_id) const {

  Action position_id = ActionToPosition(action_id);

  return absl::StrCat(StateToString(ActionToMarker(action_id)), "(",
                      position_id / kNumCols, ",", position_id % kNumCols, ")");
}

bool TicTacToeSuperGameBState::HasLine(Player player) const {
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

bool TicTacToeSuperGameBState::IsFull() const { return num_moves_ == kNumCells; }

TicTacToeSuperGameBState::TicTacToeSuperGameBState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string TicTacToeSuperGameBState::ToString() const {
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

bool TicTacToeSuperGameBState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> TicTacToeSuperGameBState::Returns() const {
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

std::string TicTacToeSuperGameBState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TicTacToeSuperGameBState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TicTacToeSuperGameBState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void TicTacToeSuperGameBState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> TicTacToeSuperGameBState::Clone() const {
  return std::unique_ptr<State>(new TicTacToeSuperGameBState(*this));
}

TicTacToeSuperGameBGame::TicTacToeSuperGameBGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace tic_tac_toe_supergame_B
}  // namespace open_spiel
