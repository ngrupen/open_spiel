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

#include "open_spiel/games/tic_tac_toe_5x5.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace tic_tac_toe_5x5 {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"tic_tac_toe_5x5",
    /*long_name=*/"Tic Tac Toe 5x5",
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
  return std::shared_ptr<const Game>(new TicTacToe5x5Game(params));
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

void TicTacToe5x5State::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
  board_[move] = PlayerToState(CurrentPlayer());
  if (HasLine(current_player_)) {
    outcome_ = current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
}

std::vector<Action> TicTacToe5x5State::LegalActions() const {
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

std::vector<Action> TicTacToe5x5State::OriginalLegalActions(Player player)  {
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

std::string TicTacToe5x5State::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

bool TicTacToe5x5State::HasLine(Player player) const {
  CellState c = PlayerToState(player);
  return (board_[0] == c && board_[1] == c && board_[2] == c && board_[3] == c && board_[4] == c) ||
         (board_[5] == c && board_[6] == c && board_[7] == c && board_[8] == c && board_[9] == c) ||
         (board_[10] == c && board_[11] == c && board_[12] == c && board_[13] == c && board_[14] == c) ||
         (board_[15] == c && board_[16] == c && board_[17] == c && board_[18] == c && board_[19] == c) ||
         (board_[20] == c && board_[21] == c && board_[22] == c && board_[23] == c && board_[24] == c) ||
         (board_[0] == c && board_[5] == c && board_[10] == c && board_[15] == c && board_[20] == c) ||
         (board_[1] == c && board_[6] == c && board_[11] == c && board_[16] == c && board_[21] == c) ||
         (board_[2] == c && board_[7] == c && board_[12] == c && board_[17] == c && board_[22] == c) ||
         (board_[3] == c && board_[8] == c && board_[13] == c && board_[18] == c && board_[23] == c) ||
         (board_[4] == c && board_[9] == c && board_[14] == c && board_[19] == c && board_[24] == c) ||
         (board_[0] == c && board_[6] == c && board_[12] == c && board_[18] == c && board_[24] == c) ||
         (board_[4] == c && board_[8] == c && board_[12] == c && board_[16] == c && board_[20] == c);
}

bool TicTacToe5x5State::IsFull() const { return num_moves_ == kNumCells; }

TicTacToe5x5State::TicTacToe5x5State(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string TicTacToe5x5State::ToString() const {
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

std::string TicTacToe5x5State::GetIDString() {
  std::string id_str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&id_str, StateToString(BoardAt(r, c)));
    }
  }
  return id_str;
}

bool TicTacToe5x5State::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> TicTacToe5x5State::Returns() const {
  if (HasLine(Player{0})) {
    return {1.0, -1.0};
  } else if (HasLine(Player{1})) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string TicTacToe5x5State::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TicTacToe5x5State::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TicTacToe5x5State::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    // std::cerr << "Cell: " << cell << ", Value: " << board_[cell] << ", Value (int): " << static_cast<int>(board_[cell]) << std::endl;
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }

  // print out each dimension of tensor view   
//   for (int idx = 0; idx < 3; idx++) {
    // for (int cell = 0; cell < kNumCells; cell++) {
    //   std::cerr << "idx: " << idx << ", cell: " << cell << ", view: " << view[{static_cast<int>(board_[idx]), cell}] << std::endl;
    //   std::cerr << "idx: " << idx << ", cell: " << cell << ", view: " << view[{idx, cell}] << std::endl;
    
    // }
    // std::cerr << "----" << std::endl;
//   }
}

void TicTacToe5x5State::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> TicTacToe5x5State::Clone() const {
  return std::unique_ptr<State>(new TicTacToe5x5State(*this));
}

TicTacToe5x5Game::TicTacToe5x5Game(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace tic_tac_toe
}  // namespace open_spiel
