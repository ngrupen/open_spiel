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

#ifndef OPEN_SPIEL_GAMES_BD_MINES_H_
#define OPEN_SPIEL_GAMES_BD_MINES_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "open_spiel/spiel.h"

// A simple jeopardy dice game that includes chance nodes.
// See http://cs.gettysburg.edu/projects/pig/index.html for details.
// Also https://en.wikipedia.org/wiki/Pig_(dice_game)
//
// Parameters:
//     "diceoutcomes"  int    number of outcomes of the dice  (default = 6)
//     "horizon"       int    max number of moves before draw (default = 1000)
//     "players"       int    number of players               (default = 2)
//     "winscore"      int    number of point needed to win   (default = 100)

namespace open_spiel {
namespace bd_mines {
// Cell types supported from Boulderdash/Emerald Mines
enum class HiddenCellType {
  kNull = -1,
  kRockford = 0,
  kEmpty = 1,
  kDirt = 2,
  kBoulder = 3,
  kBoulderFalling = 4,
  KDiamond = 5,
  kDiamondFalling = 6,
  kExitClosed = 7,
  kExitOpen = 8,
  kRockfordInExit = 9,
  kFireflyUp = 10,
  kFireflyLeft = 11,
  kFireflyDown = 12,
  kFireflyRight = 13,
  kButterflyUp = 14,
  kButterflyLeft = 15,
  kButterflyDown = 16,
  kButterflyRight = 17,
  kWallBrick = 18,
  kWallSteel = 19,
  kWallMagicDormant = 20,
  kWallMagicOn = 21,
  kWallMagicExpired = 22,
  kAmoeba = 23,
  kExplosionDiamond = 24,
  kExplosionBoulder = 25,
  kExplosionEmpty = 26
};

// Cell types which are observable
enum class VisibleCellType {
  kNull = -1,
  kRockford = 0,
  kEmpty = 1,
  kDirt = 2,
  kBoulder = 3,
  KDiamond = 4,
  kExitClosed = 5,
  kExitOpen = 6,
  kRockfordInExit = 7,
  kFirefly = 8,
  kButterfly = 9,
  kWallBrick = 10,
  kWallSteel = 11,
  kWallMagicOff = 12,
  kWallMagicOn = 13,
  kAmoeba = 14,
  kExplosion = 15,
};

constexpr int kNumVisibleCellType = 16;

// Directions the interactions take place
enum Directions {
  kNone = 0, kUp = 1, kRight = 2, kDown = 3, kLeft = 4, kUpRight = 5, 
  kDownRight = 6, kDownLeft = 7, kUpLeft = 8
};

constexpr int kNumDirections = 9;
constexpr int kNumActions = 5;

// Element entities, along with properties
struct Element {
  HiddenCellType cell_type;
  VisibleCellType visible_type;
  int properties;
  char id;
  bool has_updated;

  Element() : 
    cell_type(HiddenCellType::kNull), visible_type(VisibleCellType::kNull), properties(0), 
    id(0), has_updated(false) {}

  Element(HiddenCellType cell_type, VisibleCellType visible_type, int properties, char id) : 
    cell_type(cell_type), visible_type(visible_type), properties(properties), 
    id(id), has_updated(false) {}

  bool operator==(const Element & rhs) const {
    return this->cell_type == rhs.cell_type;
  }

  bool operator!=(const Element & rhs) const {
    return this->cell_type != rhs.cell_type;
  }
};

const Element kNullElement = {HiddenCellType::kNull, VisibleCellType::kNull, -1, 0};

struct Grid {
  int num_rows;
  int num_cols;
  std::vector<Element> elements;
};

inline constexpr char kDefaultGrid[] =
    "40,22,1280,12\n"
    "19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19\n"
    "19,02,02,02,02,02,02,01,02,02,05,02,03,01,02,02,02,02,02,03,02,03,02,02,02,02,02,02,02,01,02,02,02,02,03,02,02,02,02,19\n"
    "19,01,03,00,03,02,02,02,02,02,02,01,02,02,02,02,02,02,02,02,02,03,05,02,02,03,02,02,02,02,01,02,02,02,02,02,01,02,02,19\n"
    "19,02,02,02,02,02,02,02,02,02,02,01,02,02,03,02,02,02,02,02,03,02,03,02,02,03,02,02,02,02,02,02,02,02,03,02,02,02,02,19\n"
    "19,03,02,03,03,02,02,02,02,02,02,02,02,02,03,02,02,02,02,02,02,03,02,02,03,02,02,02,02,03,02,02,02,03,02,02,02,02,02,19\n"
    "19,03,02,01,03,02,02,02,02,02,02,02,02,02,01,03,02,02,03,02,02,02,02,02,02,02,02,03,02,02,02,02,02,02,03,02,03,03,02,19\n"
    "19,02,02,02,01,02,02,03,02,02,02,02,02,02,02,02,03,02,02,02,02,02,03,02,01,03,02,02,02,02,02,02,02,02,03,02,03,03,02,19\n"
    "19,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,02,02,02,03,02,02,03,02,19\n"
    "19,02,01,02,02,02,03,02,02,05,02,01,02,02,03,02,03,02,02,02,02,02,02,02,02,02,02,05,02,03,05,02,02,02,02,02,02,01,02,19\n"
    "19,02,02,05,02,02,02,02,02,03,02,02,02,02,02,01,02,02,02,02,02,02,02,02,03,03,01,03,02,02,03,02,02,02,02,03,02,02,02,19\n"
    "19,02,02,02,03,02,02,03,02,03,02,02,02,02,02,02,02,02,02,02,02,02,02,02,03,01,02,03,02,02,03,02,02,02,02,02,02,02,02,19\n"
    "19,02,03,02,02,02,02,02,03,02,02,02,02,02,02,02,02,03,03,03,02,02,02,02,02,02,02,03,02,02,01,02,05,02,02,02,02,03,02,19\n"
    "19,02,05,02,02,01,02,02,03,02,01,01,02,02,02,02,02,03,02,03,05,02,02,05,02,02,02,02,03,02,02,02,03,02,02,05,02,01,02,19\n"
    "19,02,01,03,02,02,02,02,02,02,02,02,02,02,02,02,02,02,03,01,03,02,02,03,02,02,02,02,02,02,02,02,05,02,02,02,02,02,03,19\n"
    "19,02,02,02,02,02,02,02,02,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,19\n"
    "19,01,03,02,02,02,02,02,02,02,02,02,03,02,02,02,05,02,02,02,02,03,02,02,02,02,02,03,02,02,02,03,02,02,02,02,02,02,02,19\n"
    "19,01,03,02,02,02,02,02,02,02,02,02,01,03,02,02,03,02,02,02,02,02,02,02,02,03,02,02,02,02,02,02,03,02,03,03,02,02,07,19\n"
    "19,02,01,02,02,03,02,02,02,02,02,02,02,02,03,02,02,02,02,02,03,02,01,01,02,02,02,02,05,02,02,02,03,02,03,03,02,02,02,19\n"
    "19,02,02,02,02,03,05,02,02,03,02,02,02,02,02,02,02,02,03,02,02,02,02,02,02,03,02,03,05,02,02,02,02,02,02,03,02,02,02,19\n"
    "19,02,02,02,01,02,02,03,02,01,02,02,03,02,03,03,02,02,02,02,02,02,02,02,02,03,02,03,05,02,02,02,02,02,02,01,02,02,03,19\n"
    "19,02,05,02,02,02,02,01,02,02,02,02,02,01,02,02,02,02,02,02,02,02,02,01,02,03,02,02,03,02,02,02,02,03,02,02,02,03,02,19\n"
    "19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19";

class BDMinesState : public State {
 public:
  BDMinesState(const BDMinesState&) = default;
  BDMinesState(std::shared_ptr<const Game> game, int steps_remaining, int magic_wall_steps,
               bool magic_active, int amoeba_max_size, int amoeba_size, Element amoeba_swap,
               bool amoeba_enclosed, int gems_required, int gems_collected, int current_reward,
               int sum_reward, Grid grid, int rng_seed) : 
      State(game),
      steps_remaining_(steps_remaining),
      magic_wall_steps_(magic_wall_steps),
      magic_active_(magic_active),
      amoeba_max_size_(amoeba_max_size),
      amoeba_size_(amoeba_size),
      amoeba_swap_(amoeba_swap),
      amoeba_enclosed_(amoeba_enclosed),
      gems_required_(gems_required),
      gems_collected_(gems_collected),
      current_reward_(current_reward),
      sum_reward_(sum_reward),
      grid_(grid),
      rng_(rng_seed) {}

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::string ObservationString(Player player) const override;
  std::string Serialize() const override;
  void ObservationTensor(Player player,
                         std::vector<double>* values) const override;

  std::unique_ptr<State> Clone() const override;

  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  int IndexFromAction(int index,  int action) const;
  bool InBounds(int index, int action=Directions::kNone) const;
  bool IsType(int index, Element element, int action=Directions::kNone) const;
  bool HasProperty(int index, int property, int action=Directions::kNone) const;
  void MoveItem(int index, int action);
  void SetItem(int index, Element element, int action=Directions::kNone);
  Element GetItem(int index, int action=Directions::kNone) const;
  bool IsTypeAdjacent(int index, Element element) const;

  bool CanRollLeft(int index) const;
  bool CanRollRight(int index) const;
  void RollLeft(int index, Element element);
  void RollRight(int index, Element element);
  void Push(int index, int action);
  void MoveThroughMagic(int index, Element element);
  void Explode(int index, Element element, int action=Directions::kNone);

  void UpdateBoulder(int index);
  void UpdateBoulderFalling(int index);
  void UpdateDiamond(int index);
  void UpdateDiamondFalling(int index);
  void UpdateExit(int index);
  void UpdateRockford(int index, int action);
  void UpdateFirefly(int index, int action);
  void UpdateButterfly(int index, int action);
  void UpdateMagicWall(int index);
  void UpdateAmoeba(int index);
  void UpdateExplosions(int index);

  void StartScan();
  void EndScan();

  int steps_remaining_;   // Max steps before game over
  int magic_wall_steps_;  //steps before magic wall expire (after active)
  bool magic_active_;     // flag for magic wall state
  int amoeba_max_size_;   // size before amoebas collapse
  int amoeba_size_;       // current number of amoebas
  Element amoeba_swap_;   // Element which amoebas swap to
  bool amoeba_enclosed_;  // internal flag to check if amoeba trapped
  int gems_required_;     // gems required to open exit
  int gems_collected_;    // gems collected thus far
  double current_reward_; // reset at every step
  double sum_reward_;     // cumulative reward
  Grid grid_;             // grid representing elements/positions
  mutable std::mt19937 rng_;      // Internal rng

  // Initialize to bad/invalid values. Use open_spiel::NewInitialState()
  Player cur_player_ = -1;  // Player to play.
};

class BDMinesGame : public Game {
 public:
  explicit BDMinesGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new BDMinesState(shared_from_this(), max_steps_, magic_wall_steps_, false, amoeba_max_size_,
                         0, kNullElement, true, gems_required_, 0, 0, 0, 
                         grid_, ++rng_seed_));
  }
  int MaxGameLength() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new BDMinesGame(*this));
  }
  std::vector<int> ObservationTensorShape() const override;
  std::unique_ptr<State> DeserializeState(const std::string& str) const override;

protected:
  Grid ParseGrid(const std::string& grid_string);

 private:
  int max_steps_;         // Max steps before game over
  int magic_wall_steps_;  //steps before magic wall expire (after active)
  int amoeba_max_size_;   // size before amoebas collapse
  mutable int rng_seed_;  // Seed for stochastic element transitions
  Grid grid_;             // grid representing elements/positions
  int gems_required_;     // gems required to open exit
};

}  // namespace bd_mines
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BD_MINES_H_
