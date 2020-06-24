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

#ifndef OPEN_SPIEL_PUBLIC_STATES_H_
#define OPEN_SPIEL_PUBLIC_STATES_H_

#if !defined(BUILD_WITH_PUBLIC_STATES) || !defined(BUILD_WITH_EIGEN)
#error "To use public states you must enable building with both " \
       "BUILD_WITH_PUBLIC_STATES=ON and BUILD_WITH_EIGEN=ON env vars."
#endif

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/eigen/pyeig.h"

#include "open_spiel/abseil-cpp/absl/random/bit_gen_ref.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/random.h"

// This files specifies the public state API for OpenSpiel.
// It is an imperfect-recall abstraction for Factored-Observation Games [1].
// Many of the decisions are described in the README documentation.
//
// [1] https://arxiv.org/abs/1906.11110

namespace open_spiel {
namespace public_states {

// Static information for a game. This will determine what algorithms,
// solution concepts and automatic consistency checks are applicable.
struct GameWithPublicStatesType {
    // A short name with no spaces that uniquely identifies the game, e.g.
    // "msoccer". This is the key used to distinguish games. It must be the same
    // as GameType::short_name of the underlying Base API.
    std::string short_name;

    // Provides methods that compute values needed to run counter-factual
    // regret minimization: beliefs and cf. values.
    bool provides_cfr_computation;

    // Does the implementation provide IsStateCompatible() implementations?
    // These are useful for automatic consistency checks with Base API.
    //
    // They provide a way of comparing imperfect-recall public / private
    // informations with perfect recall public-observation histories /
    // action private-observation histories, as these are uniquely defined for
    // each State.
    bool provides_state_compatibility_check;
};


// An abstract class that represents a private information in the game.
//
// This is an imperfect-recall variant of private information. This means there
// might be multiple Action-PrivateObservation histories that yield the same
// private information.
//
// The private information does not contain any piece of public information!
class PrivateInformation {
 public:
  explicit PrivateInformation(std::shared_ptr<const Game> game);
  PrivateInformation(const PrivateInformation &) = default;
  virtual ~PrivateInformation() = default;

  // The player that owns this private information.
  virtual Player GetPlayer() const {
    SpielFatalError("GetPlayer() is not implemented.");
  }

  // A number that uniquely identifies position of this private information
  // within a belief vector.
  //
  // Equality of PrivateInformation implies the same BeliefIndex().
  virtual unsigned int BeliefIndex() const {
    SpielFatalError("BeliefIndex() is not implemented.");
  }

  // A number that uniquely identifies position of this private information
  // within a neural network input.
  //
  // Equality of PrivateInformation implies the same NetworkIndex().
  virtual unsigned int NetworkIndex() const {
    SpielFatalError("NetworkIndex() is not implemented.");
  }

  // Return representation for neural networks.
  // TODO(sustr): tensor layouts.
  // Equality of PrivateInformation implies they have the same ToTensor()
  virtual std::vector<double> ToTensor() const {
    SpielFatalError("ToTensor() is not implemented.");
  }

  // Can State produce this private information?
  // 
  // Implementing this method is optional, but highly recommended, as it
  // helps with testing consistency of the implementation with Base API.
  //
  // See also GameWithPublicStatesType::provides_state_compatibility_check
  virtual bool IsStateCompatible(const State &) const {
    SpielFatalError("IsStateCompatible() is not implemented.");
  }

  // A human-readable string representation.
  // Equality of PrivateInformation implies they have the same ToString()
  virtual std::string ToString() const {
    SpielFatalError("ToString() is not implemented.");
  }

  virtual std::unique_ptr<PrivateInformation> Clone() const {
    SpielFatalError("Clone() is not implemented.");
  }

  // Serializes a private information into a string.
  //
  // If overridden, this must be the inverse of
  // GameWithPublicStates::DeserializePrivateInformation
  //
  // Two PrivateInformations are equal if and only if they have the same
  // Serialize() outputs.
  virtual std::string Serialize() const {
    SpielFatalError("Serialize() is not implemented.");
  }

  // Compare whether the other private information is equal.
  virtual bool operator==(const PrivateInformation &other) const {
    SpielFatalError("operator==() is not implemented.");
  }

  bool operator!=(const PrivateInformation &other) const {
    return !operator==(other);
  }

  // Get the game object that generated this state.
  std::shared_ptr<const Game> GetGame() const { return game_; }

 protected:
  // A pointer to the game that created this public information.
  const std::shared_ptr<const Game> game_;
};


// An abstract class that represents a public information in the game.
//
// This is an imperfect-recall variant of public information. This means there
// might be multiple Action-PublicObservation histories that yield the same
// public information.
//
// The public information does not contain any piece of private information!
class PublicInformation {
 public:
  explicit PublicInformation(std::shared_ptr<const Game> game);
  PublicInformation(const PublicInformation &) = default;
  virtual ~PublicInformation() = default;

  // Return representation for neural networks.
  // TODO(sustr): tensor layouts.
  // Equality of PublicInformation implies they have the same ToTensor()
  virtual std::vector<double> ToTensor() const {
    SpielFatalError("ToTensor() is not implemented.");
  }

  // A human-readable string representation.
  // Equality of PublicInformation implies they have the same ToString()
  virtual std::string ToString() const {
    SpielFatalError("ToString() is not implemented.");
  }

  // Can State produce this public information?
  //
  // Implementing this method is optional, but highly recommended, as it
  // helps with testing consistency of the implementation with Base API.
  //
  // See also GameWithPublicStatesType::provides_state_compatibility_check
  virtual bool IsStateCompatible(const State &) const {
    SpielFatalError("IsStateCompatible() is not implemented.");
  }

  virtual std::unique_ptr<PublicInformation> Clone() const {
    SpielFatalError("Clone() is not implemented.");
  }

  // Serializes a public information into a string.
  //
  // If overridden, this must be the inverse of
  // GameWithPublicStates::DeserializePublicInformation
  //
  // Two PublicInformations are equal if and only if they have the same
  // Serialize() outputs.
  virtual std::string Serialize() const {
    SpielFatalError("Serialize() is not implemented.");
  }

  // Compare whether the other public information is equal.
  virtual bool operator==(const PublicInformation &other) const {
    SpielFatalError("operator==() is not implemented.");
  };

  bool operator!=(const PublicInformation &other) const {
    return !operator==(other);
  };

  // Get the game object that generated this state.
  std::shared_ptr<const Game> GetGame() const { return game_; }

 protected:
  // A pointer to the game that created this public information.
  const std::shared_ptr<const Game> game_;
};

// An edge in the public tree. This is the same as output of
// State::PublicObservationString()
using PublicTransition = std::string;

// A container for counter-factual values for each private state (private
// information) within a public state. The values are always accompanied by the
// owning player. The values must be always within the range of game's
// Max/Min Returns.
struct CfPrivValues {
    const Player player;
    VectorXd cfvs;
};

// A container for counter-factual action-values for each action of a private
// state (private information). The values are always accompanied by the owning
// player. The values must be always within the range of game's Max/Min Returns.
struct CfActionValues {
    const Player player;
    VectorXd cfavs;
};

// A container for beliefs (prob. distribution) for each private information
// within a public state. The values are always accompanied by the owning
// player. The beliefs must sum to 1.
struct Beliefs {
    const Player player;
    VectorXd beliefs;
};

// A public state is perfect recall - it corresponds to an object specified by
// public-observation history and provides methods on top of it.
// It corresponds to a specific node within a public tree.
class PublicState {
 public:
  explicit PublicState(std::shared_ptr<const Game> game);
  PublicState(const PublicState &) = default;
  virtual ~PublicState() = default;

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Perspectives over the public state.">
  // ---------------------------------------------------------------------------

  // Return the public observation history.
  const std::vector<PublicTransition> &GetPublicObservationHistory() const {
    return pub_obs_history_;
  }

  virtual const PublicInformation &GetPublicInformation() const {
    SpielFatalError("GetPublicInformation() is not implemented.");
  }

  // Return numbers of the private informations consistent with the
  // public information (for each player).
  virtual std::vector<int> NumDistinctPrivateInformations() const {
    SpielFatalError(
        "NumDistinctPrivateInformations() is not implemented.");
  }

  // Returns all the possible private informations for requested player that
  // are possible for the public information of this public state.
  // Their ordering within the returned vector must be consistent with
  // their BeliefIndex, the first element having BeliefIndex = 0 and the last
  // element having BeliefIndex = returned_list.size()-1
  virtual std::vector<PrivateInformation> GetPrivateInformations(
      Player) const {
    SpielFatalError("GetPrivateInformations() is not implemented.");
  }

  // Return all States that are consistent with this public state in the sense
  // that they have the same public observation history. However, there may
  // be an exponential number of them, given that we are doing imperfect
  // recall abstraction. Therefore, return a minimally sized set of these
  // states such that there is no other state that is isomorphic to any
  // of them.
  virtual std::vector<State> GetPublicSet() const {
    SpielFatalError("GetPublicSet() is not implemented.");
  }

  // Return an information state string description for this public state
  // + private information.
  virtual std::string GetInformationState(const PrivateInformation &) const {
    SpielFatalError("GetPublicSet() is not implemented.");
  }

  // Return all states that are consistent with the player’s private
  // information and public state.
  virtual std::vector<State> GetInformationSet(
      const PrivateInformation &) const {
    SpielFatalError("GetInformationSet() is not implemented.");
  }

  // Return State that corresponds to the combination of player’s private
  // informations and this public state.
  virtual std::unique_ptr<State> GetWorldState(
      const std::vector<PrivateInformation *> &) const {
    SpielFatalError("GetWorldState() is not implemented.");
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Fetch a random subset from a perspective">
  // ---------------------------------------------------------------------------

  virtual std::unique_ptr<State> ResampleFromPublicSet(Random *) const {
    SpielFatalError("ResampleFromPublicSet() is not implemented.");
  }

  virtual std::unique_ptr<State> ResampleFromInformationSet(Random *) const {
    SpielFatalError("ResampleFromPublicSet() is not implemented.");
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Traversal of public state">
  // ---------------------------------------------------------------------------

  // Updates the public state when a public-tree action (transition) is made.
  // Games should implement DoApplyPublicTransition.
  //
  // Mutates public state!
  void ApplyPublicTransition(const PublicTransition &transition) {
    DoApplyPublicTransition(transition);
    pub_obs_history_.push_back(transition);
  }

  // Updates the public state when a world-level action is made.
  // Games should implement DoApplyStateAction. This should work also for
  // simultaneous-move games.
  //
  // Mutates public state!
  virtual PublicTransition ApplyStateAction(
      const std::vector<PrivateInformation *> &privates, Action action) {
    PublicTransition transition = DoApplyStateAction(privates, action);
    pub_obs_history_.push_back(transition);
    return transition;
  }

  virtual std::unique_ptr<PublicState> Child(const PublicTransition &) const {
    SpielFatalError("Child() is not implemented.");
  }

  virtual std::vector<PublicTransition> GetPublicTransitions() const {
    SpielFatalError("GetPublicTransitions() is not implemented.");
  }

  // For each private information of a player return its private actions.
  // Note that if the player is not acting in this public state, the returned
  // actions should be empty.
  // [PrivateInformation x Private Actions]
  virtual std::vector<std::vector<Action>> GetPrivateActions(Player) const {
    SpielFatalError("GetPrivateActions() is not implemented.");
  }

  // Undoes the last transition, which must be supplied. This is a method
  // to get a parent public state.
  //
  // Mutates public state!
  virtual void UndoTransition(const PublicTransition &) {
    SpielFatalError("UndoTransition() is not implemented.");
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Public state types">
  // ---------------------------------------------------------------------------

  // Is this a chance public state?
  virtual bool IsChance() const {
    SpielFatalError("IsChance() is not implemented.");
  }

  // Is this public state terminal?
  virtual bool IsTerminal() const {
    SpielFatalError("IsTerminal() is not implemented.");
  }

  // Is this a player public state?
  virtual bool IsPlayer() const {
    SpielFatalError("IsPlayer() is not implemented.");
  }

  // Collection of currently acting players, if this is a player public state.
  // There is only one player acting within a public state for
  // GameType::Dynamics::kSequential.
  virtual std::vector<Player> ActingPlayers() const {
    SpielFatalError("IsPlayer() is not implemented.");
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Terminals">
  // ---------------------------------------------------------------------------

  // Return utility of a terminal world state for each player.
  virtual std::vector<double> TerminalReturns(
      const std::vector<PrivateInformation *> &privates) const {
    std::unique_ptr<State> terminal_state = GetWorldState(privates);
    SPIEL_CHECK_TRUE(terminal_state->IsTerminal());
    return terminal_state->Returns();
  }

  // Return the matrix of terminal utilities.
  // Available only for two-player zero-sum games.
  virtual MatrixXd TerminalUtilities() const {
    SpielFatalError("TerminalUtilities() is not implemented.");
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="CFR-related computations">
  // ---------------------------------------------------------------------------

  // Beliefs -------------------------------------------------------------------

  // Compute beliefs of a player if it was going to transition to a child
  // public state. The belief is updated using the private strategy of the
  // player if needed. This function works also for players who do not act in
  // this public state, they just supply an empty strategy. All players beliefs
  // should be updated for the chance transitions.
  //
  // Strategy of the player: [private_states x private_actions]
  // Each element of the vector should be a valid probability distribution.
  virtual Beliefs ComputeBeliefs(
      const PublicTransition &,
      const std::vector<VectorXd> &strategy,
      const Beliefs &) {
    SpielFatalError("UpdatePlayerBeliefs() is not implemented.");
  }

  // Counter-factual values ----------------------------------------------------

  // Return the counter-factual values for terminals for each player,
  // given their beliefs.
  virtual std::vector<CfPrivValues> TerminalCfValues(
      const std::vector<Beliefs> &) const {
    SpielFatalError("TerminalCfValues() is not implemented.");
  }

  // Compute counter-factual values of private states (information states)
  // corresponding to the imperfect-recall private informations.
  //
  // This is a vectorized version for computation of the second term
  // from Regret Matching procedure:
  //
  //   r(I,a) = v(I,a) + \sum \sigma(I,a) v(I,a)
  //
  // We are provided the children cf. action-values and the player policies
  // to reach them. The action-values have the same size as the privates
  // strategy:
  //
  // children_values: [private_state I  x  cf. action values]
  // children_policy: [private_state I  x  policy for each action]
  // Returned values: [cf. value per private_state I]
  virtual CfPrivValues ComputeCfPrivValues(
      const std::vector<CfActionValues> &children_values,
      const std::vector<VectorXd> &children_policy) const {
    SpielFatalError("ComputeCfPrivValues() is not implemented.");
  }

  // Compute counter-factual action-values (i.e. values when the player follows
  // each private action with 100% probability).
  //
  // This is a vectorized version for computation of the first term
  // from Regret Matching procedure:
  //
  //   r(I,a) = v(I,a) + \sum \sigma(I,a) v(I,a)
  //
  // Within the private tree of a player this term v(I,a) can be computed as sum
  // of the child infosets J:
  //
  //   v(I,a) = \sum v(J)
  //
  // Note that these child infosets J belong to the next public state and are
  // designed to be retrieved from the neural network.
  //
  // We are provided these children cf. values and we return cf. action-values:
  //
  // children_values: [private_state I  x  cf. values for child infosets J]
  // Returned values: [private_state I  x  cf. action-value for each 'a' of 'I']
  virtual std::vector<CfActionValues> ComputeCfActionValues(
      const std::vector<CfPrivValues> &children_values) const {
    SpielFatalError("ComputeCfActionValues() is not implemented.");
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Neural networks">
  // ---------------------------------------------------------------------------

  // Return appropriate representation of this public state.
  // TODO(sustr): tensor layouts.
  // This function uses PrivateInformation::NetworkIndex()
  // to place features of private informations.
  virtual std::vector<double> ToTensor() const {
    SpielFatalError("ToTensor() is not implemented.");
    // TODO(sustr): default implementation concat public + privates?
  }

  // TODO(sustr): tensor layouts.
  // This function uses PrivateInformation::NetworkIndex() to place beliefs.
  virtual std::vector<double> ToTensor(const std::vector<Beliefs> &) const {
    SpielFatalError("ToTensor() is not implemented.");
    // TODO(sustr): default implementation concat ToTensor() + beliefs?
  }

  // </editor-fold>

  // ---------------------------------------------------------------------------
  // <editor-fold desc="Miscellaneous">
  // ---------------------------------------------------------------------------

  // Human readable description of the public state.
  virtual std::string ToString() const {
    SpielFatalError("ToString() is not implemented.");
  }

  // Depth of the public state within the public tree.
  virtual int GetDepth() const {
    return pub_obs_history_.size();
  }

  virtual std::unique_ptr<PublicState> Clone() const {
    SpielFatalError("Clone() is not implemented.");
  }

  // Serializes a public state into a string.
  //
  // If overridden, this must be the inverse of
  // GameWithPublicStates::DeserializePublicState
  virtual std::string Serialize() const {
    SpielFatalError("Serialize() is not implemented.");
  }

  // Get the game object that generated this state.
  std::shared_ptr<const Game> GetGame() const { return game_; }

  // Compare if the public state has exactly the same public observation
  // history. This is not the same as comparing two PublicInformations!
  virtual bool operator==(const PublicState &other) const {
    SpielFatalError("operator==() is not implemented.");
  }

  bool operator!=(const PublicState &other) const {
    return !operator==(other);
  }

  // </editor-fold>

 protected:
  // See ApplyPublicTransition.
  // Mutates public state!
  virtual void DoApplyPublicTransition(const PublicTransition&) {
    SpielFatalError("DoApplyPublicTransition() is not implemented.");
  }
  // See ApplyStateAction.
  // Mutates public state!
  virtual PublicTransition DoApplyStateAction(
      const std::vector<PrivateInformation *> &, Action) {
    SpielFatalError("DoApplyStateAction() is not implemented.");
  }

  // Public observations received so far.
  std::vector<PublicTransition> pub_obs_history_;

  // A pointer to the game that created this public state.
  const std::shared_ptr<const Game> game_;
};

// An abstract game class that provides methods for constructing
// public state API objects and asking for properties of the public tree.
class GameWithPublicStates {
 public:
  GameWithPublicStates(std::shared_ptr<const Game> game)
      : game_(std::move(game)) {};
  GameWithPublicStates(const GameWithPublicStates &) = default;
  virtual ~GameWithPublicStates() = default;

  // Create a new initial public state, that is a root of the public tree.
  virtual std::unique_ptr<PublicState> NewInitialPublicState() const {
    SpielFatalError("NewInitialPublicState() is not implemented.");
  }

  // Create beliefs that players have for the root public state.
  virtual std::unique_ptr<PublicState> NewInitialBeliefs() const {
    SpielFatalError("NewInitialBeliefs() is not implemented.");
  }

  // Provide information about the maximum number of distinct private
  // informations in any public state in the game. Note that this should not
  // be an arbitrary upper bound, but indeed the maximum number, because
  // it serves to specify the sizes of neural network inputs. Some algorithms
  // will likely use it to preallocate memory.
  //
  // Example: in HUNL Poker players receive 2 cards from the pile of 52 cards,
  // which makes it 52 * 51 / 2 = 1326 for each player.
  virtual std::vector<int> MaxDistinctPrivateInformationsCount() const {
    SpielFatalError(
        "MaxDistinctPrivateInformationsCount() is not implemented.");
  }

  // Returns a newly allocated private information built from a string.
  // Caller takes ownership of the object.
  //
  // If this method is overridden, then it should be the inverse of
  // PrivateInformation::Serialize (i.e. that method should also be overridden).
  virtual std::unique_ptr<PrivateInformation>
  DeserializePrivateInformation() const {
    SpielFatalError("DeserializePrivateInformation() is not implemented.");
  }

  // Returns a newly allocated public information built from a string.
  // Caller takes ownership of the object.
  //
  // If this method is overridden, then it should be the inverse of
  // PublicInformation::Serialize (i.e. that method should also be overridden).
  virtual std::unique_ptr<PublicInformation>
  DeserializePublicInformation() const {
    SpielFatalError("DeserializePublicInformation() is not implemented.");
  }

  // Returns a newly allocated public state built from a string.
  // Caller takes ownership of the object.
  //
  // If this method is overridden, then it should be the inverse of
  // PublicState::Serialize (i.e. that method should also be overridden).
  virtual std::unique_ptr<PublicState> DeserializePublicState() const {
    SpielFatalError("DeserializePublicState() is not implemented.");
  }

  std::shared_ptr<const Game> game_;
};

#define REGISTER_SPIEL_GAME_WITH_PUBLIC_STATE_API(info, factory) \
  GameWithPublicStatesRegisterer CONCAT(game, __COUNTER__)(info, factory);

class GameWithPublicStatesRegisterer {
 public:
  using CreateFunc = std::function<
      std::shared_ptr<const GameWithPublicStates>(
          std::shared_ptr<const Game>)>;

  GameWithPublicStatesRegisterer(
      const GameWithPublicStatesType &game_type, CreateFunc creator);

  static std::shared_ptr<const GameWithPublicStates> CreateByName(
      const std::string &short_name, const GameParameters &params);
  static std::shared_ptr<const GameWithPublicStates> CreateByGame(
      std::shared_ptr<const Game> base_game);

  static std::vector<std::string> RegisteredNames();
  static std::vector<GameWithPublicStatesType> RegisteredGames();
  static bool IsValidName(const std::string &short_name);
  static void RegisterGame(
      const GameWithPublicStatesType &game_type, CreateFunc creator);

 private:
  // Returns a "global" map of registrations (i.e. an object that lives from
  // initialization to the end of the program). Note that we do not just use
  // a static data member, as we want the map to be initialized before first
  // use.
  static std::map<std::string, std::pair<GameWithPublicStatesType, CreateFunc>>
  &factories() {
    static std::map<
        std::string, std::pair<GameWithPublicStatesType, CreateFunc>> impl;
    return impl;
  }
};

// Returns true if the game is registered with public state API,
// false otherwise.
bool IsGameRegisteredWithPublicStates(const std::string &short_name);

// Returns a list of games that have public state API.
std::vector<std::string> RegisteredGamesWithPublicStates();

// Returns a new game object from the specified string, which is the short
// name plus optional parameters, e.g. "go(komi=4.5,board_size=19)"
std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    const std::string &game_string);

// Returns a new game object with the specified parameters.
std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    const std::string &short_name,
    const GameParameters &params);

// Returns a new game object with the specified parameters; reads the name
// of the game from the 'name' parameter (which is not passed to the game
// implementation).
std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    GameParameters params);

// Returns a new game object from the underlying Base API game object.
std::shared_ptr<const GameWithPublicStates> LoadGameWithPublicStates(
    std::shared_ptr<const Game> game);

std::string SerializeGameWithPublicState(
    const GameWithPublicStates &game, const PublicState &state);

std::pair<std::shared_ptr<const GameWithPublicStates>,
          std::unique_ptr<PublicState>>
DeserializeGameWithPublicState(const std::string &serialized_state);

}  // namespace public_states
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PUBLIC_STATES_H_
