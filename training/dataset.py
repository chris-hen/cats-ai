from training.encode import encode_observation, encode_action


def collect_game_examples(game_result):
    """
    Erstellt Trainingsbeispiele aus einer Partie.

    Target:
    round_survived = 1.0, wenn der Spieler die Runde überlebt hat
    round_survived = 0.0, wenn er in dieser Runde ausgeschieden ist
    """

    history = game_result["history"]
    winner = game_result["winner"]

    examples = []

    # Für jede Runde herausfinden, wer am Ende rausgeflogen ist
    eliminated_by_round = {}

    for entry in history:
        info = entry.get("info")

        if info and info["event"] == "doubt":
            round_number = entry["round_number"]
            wrong_player = info["wrong_player"]
            eliminated_by_round[round_number] = wrong_player

    for entry in history:
        player_id = entry["player_id"]
        round_number = entry["round_number"]

        eliminated_player = eliminated_by_round.get(round_number)

        if eliminated_player is None:
            round_survived = 1.0
        else:
            round_survived = 0.0 if player_id == eliminated_player else 1.0

        examples.append({
            "observation": entry["observation"],
            "action_id": entry["action_id"],
            "player_id": player_id,
            "round_number": round_number,
            "round_survived": round_survived,
            "game_won": 1.0 if player_id == winner else 0.0,
        })

    return examples