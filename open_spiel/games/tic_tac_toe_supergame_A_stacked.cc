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

#include "open_spiel/games/tic_tac_toe_supergame_A_stacked.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace tic_tac_toe_supergame_A_stacked {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"tic_tac_toe_supergame_A_stacked",
    /*long_name=*/"Tic Tac Toe (SuperGame A stacked)",
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
  return std::shared_ptr<const Game>(new TicTacToeSuperGameAStackedGame(params));
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

void TicTacToeSuperGameAStackedState::DoApplyAction(Action move) {
  previous_board_ = board_;

  if (board_[move] == CellState::kEmpty) {
    board_[move] = PlayerToState(CurrentPlayer());
    if (HasLine(current_player_)) {
      outcome_ = current_player_;
    }
  } else {
      board_[move] = PlayerToState(CurrentPlayer());
      invalid_mover_ = current_player_;
      outcome_ = 1 - current_player_;
  }
  current_player_ = 1 - current_player_;
  num_moves_ += 1;
//   std::cerr << "Apply action player: " << current_player_ << std::endl;

//   std::string board_str;
//   for (int r = 0; r < kNumRows; ++r) {
//     for (int c = 0; c < kNumCols; ++c) {
//       absl::StrAppend(&board_str, StateToString(BoardAt(r, c)));
//     }
//     if (r < (kNumRows - 1)) {
//       absl::StrAppend(&board_str, "\n");
//     }
//   }
//   std::cerr << "current board: \n" << board_str << std::endl;
  
//   std::string prev_board_str;
//   for (int r = 0; r < kNumRows; ++r) {
//     for (int c = 0; c < kNumCols; ++c) {
//       absl::StrAppend(&prev_board_str, StateToString(previous_board_[r * kNumCols + c]));
//     }
//     if (r < (kNumRows - 1)) {
//       absl::StrAppend(&prev_board_str, "\n");
//     }
//   }
//   std::cerr << "previous board: \n" << prev_board_str << std::endl;

}

std::vector<Action> TicTacToeSuperGameAStackedState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;
  for (int cell = 0; cell < kNumCells; ++cell) {
    moves.push_back(cell);
  }
//   std::cerr << "Legal moves: " << absl::StrJoin(moves, ",") << std::endl;
  return moves;
}

std::vector<Action> TicTacToeSuperGameAStackedState::OriginalLegalActions(Player player) {
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

int TicTacToeSuperGameAStackedState::CheckValidMove(Player player, Action action) {
  // 0 = valid, 1 = invalid --> can only be invalid if piece placed on top of another piece
  int score = 1;
  std::vector<Action> real_legal_moves = OriginalLegalActions(player);

  for (int i = 0; i < real_legal_moves.size(); ++i) {
    if (action == real_legal_moves[i]) {
      score = 0;
      return score;
    }
  }
  return score;
}

std::string TicTacToeSuperGameAStackedState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

bool TicTacToeSuperGameAStackedState::HasLine(Player player) const {
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

bool TicTacToeSuperGameAStackedState::IsFull() const { return num_moves_ == kNumCells; }

TicTacToeSuperGameAStackedState::TicTacToeSuperGameAStackedState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
  std::fill(begin(previous_board_), end(previous_board_), CellState::kEmpty);
}

std::string TicTacToeSuperGameAStackedState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
//   std::cerr << "state str:\n" << str << "\n" << std::endl;
//   std::cerr << "----" << std::endl;
  return str;
}

std::string TicTacToeSuperGameAStackedState::GetIDString() {
  std::string id_str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&id_str, StateToString(BoardAt(r, c)));
    }
  }
  absl::StrAppend(&id_str, current_player_);
//   std::cerr << "state ID str:" << id_str << "\n" << std::endl;
  return id_str;
}

bool TicTacToeSuperGameAStackedState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> TicTacToeSuperGameAStackedState::Returns() const {
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

std::string TicTacToeSuperGameAStackedState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string TicTacToeSuperGameAStackedState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void TicTacToeSuperGameAStackedState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
//   std::cerr << "Cell states " << kCellStates << std::endl;
//   std::cerr << "Cells " << kNumCells << std::endl;
//   std::cerr << "Values: " << absl::StrJoin(values, ",") << std::endl;

  // piece locations
  TensorView<2> view(values, {(kCellStates+1)*2, kNumCells}, true);

  // current board
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
    // std::cerr << "Cell " << cell << std::endl;
    // std::cerr << "Value " << board_[cell] << std::endl;
  }

  // current player plane
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{3, cell}] = current_player_;
  }

  // previous board
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(previous_board_[cell])+(kCellStates+1), cell}] = 1.0;
    // std::cerr << "Cell " << cell << std::endl;
    // std::cerr << "Value " << board_[cell] << std::endl;
  }
  
  // previous player plane
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{7, cell}] = 1 - current_player_;
  }

  // print out each dimension of tensor view   
//   for (int idx = 0; idx < 8; idx++) {
//     for (int cell = 0; cell < kNumCells; cell++) {
//       std::cerr << "idx: " << idx << ", cell: " << cell << ", view: " << view[{idx, cell}] << std::endl;
    
//     }
//     std::cerr << "----" << std::endl;
//   }

//   int size = view.size();
//   const int rank = view.rank();
//   std::array<int, rank> shape = view.shape();
//   std::cerr << "observation tensor player: " << player << std::endl;
//   std::cerr << "observation tensor current player: " << current_player_ << std::endl;
//   std::cerr << "Size: " << size << std::endl;
//   std::cerr << "Rank: " << rank << std::endl;
//   std::cerr << "Shape: " << absl::StrJoin(shape, ",") << std::endl;
}

// void TicTacToeSuperGameAStackedState::ObservationTensor(Player player,
//                                        absl::Span<float> values) const {
//   SPIEL_CHECK_GE(player, 0);
//   SPIEL_CHECK_LT(player, num_players_);

//   // Treat `values` as a 2-d tensor.
//   TensorView<2> view(values, {kCellStates, kNumCells}, true);
//   for (int cell = 0; cell < kNumCells; ++cell) {
//     view[{static_cast<int>(board_[cell]), cell}] = 1.0;
//   }
// }

void TicTacToeSuperGameAStackedState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> TicTacToeSuperGameAStackedState::Clone() const {
  return std::unique_ptr<State>(new TicTacToeSuperGameAStackedState(*this));
}

TicTacToeSuperGameAStackedGame::TicTacToeSuperGameAStackedGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace tic_tac_toe_supergame_B
}  // namespace open_spiel
