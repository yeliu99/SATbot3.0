import nltk

from model.models import UserModelSession, Choice, UserModelRun, Protocol
from model.classifiers import get_emotion, get_sentence_score
import pandas as pd
import numpy as np
import random
from collections import deque
import re
import datetime
import time

nltk.download("wordnet")
from nltk.corpus import wordnet  # noqa


class ModelDecisionMaker:
    def __init__(self):

        self.PROTOCOL_TITLES = [
            "0: None",
            "1: Recalling significant early memories",
            "2: Becoming intimate with your Child",
            "3: Singing a song of affection",
            "4: Expressing love and care for the Child",
            "5: Pledging to care and support our Child",
            "6: Restoring our emotional world after our pledge",
            "7: Maintaining a loving relationship with your Child",
            "8: Creating zest for life",
            "9: Enjoying nature",
            "10: Overcoming your current negative emotions",
            "11: Overcoming past pain",
            "12: Muscle relaxation and playful face",
            "13: Laughing on your own",
            "14: Laughing with your childhood self",  # noqa
            "15: Creating your own brand of laughter",  # noqa
            "16: Learning to change your perspective",
            "17: Learning to be playful about your past pains",
            "18: Identifying our personal resentments and acting them out",  # noqa
            "19: Planning more constructive actions",
            "20: Updating our beliefs to enhance creativity",
            "21: Practicing Affirmations",
        ]

        self.TITLE_TO_PROTOCOL = {
            self.PROTOCOL_TITLES[i]: i for i in range(len(self.PROTOCOL_TITLES))
        }

        self.recent_protocols = deque(maxlen=20)
        self.reordered_protocol_questions = {}
        self.protocols_to_suggest = []

        # Goes from user id to actual value
        self.current_run_ids = {}
        self.current_protocol_ids = {}

        self.current_protocols = {}

        self.positive_protocols = [i for i in range(1, 21)]

        self.INTERNAL_PERSECUTOR_PROTOCOLS = [
            self.PROTOCOL_TITLES[15],
            self.PROTOCOL_TITLES[16],
            self.PROTOCOL_TITLES[8],
            self.PROTOCOL_TITLES[19],
        ]

        # Keys: user ids, values: dictionaries describing each choice (in list)
        # and current choice
        self.user_choices = {}

        # Keys: user ids, values: current suggested protocols
        self.suggestions = {}

        # Tracks current emotion of each user after they classify it
        self.user_emotions = {}

        self.guess_emotion_predictions = {}
        # Structure of dictionary: {question: {
        #                           model_prompt: str or list[str],
        #                           choices: {maps user response to next protocol},
        #                           protocols: {maps user response to protocols to suggest},
        #                           }, ...
        #                           }
        # This could be adapted to be part of a JSON file (would need to address
        # mapping callable functions over for parsing).

        self.users_names = {}
        self.targetA_names = {}
        self.users_feelings = {}
        self.remaining_choices = {}

        self.recent_questions = {}

        self.chosen_personas = {}
        # self.datasets = pd.read_csv('/Users/yeliu/IC/Individual_Project/code/sat3.0/model/sat.csv',
        #                        encoding='ISO-8859-1')
        self.datasets = pd.read_csv('sat.csv')

        self.QUESTIONS = {

            "ask_name": {
                "model_prompt": "Please enter your first name:",
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_name(user_id)
                },
                "protocols": {"open_text": []},
            },

            "intro_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_intro_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(
                        user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },

            "guess_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_guess_emotion(
                    user_id, app, db_session
                ),
                "choices": {
                    "Yes": {
                        "Sad": "after_classification_negative",
                        "Angry": "after_classification_negative",
                        "Anxious/Scared": "after_classification_negative",
                        "Happy/Content": "after_classification_positive",
                    },
                    "No": "check_emotion",
                },
                "protocols": {
                    "Yes": [],
                    "No": []
                },
            },

            "check_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_check_emotion(
                    user_id, app, db_session),

                "choices": {
                    "Sad": lambda user_id, db_session, curr_session, app: self.get_sad_emotion(user_id),
                    "Angry": lambda user_id, db_session, curr_session, app: self.get_angry_emotion(user_id),
                    "Anxious/Scared": lambda user_id, db_session, curr_session, app: self.get_anxious_emotion(user_id),
                    "Happy/Content": lambda user_id, db_session, curr_session, app: self.get_happy_emotion(user_id),
                    "Others, but positive feelings": lambda user_id, db_session, curr_session,
                                                            app: self.get_positive_emotion(user_id),
                    "Others, but negative feelings": lambda user_id, db_session, curr_session,
                                                            app: self.get_negative_emotion(user_id),

                },
                "protocols": {
                    "Sad": [],
                    "Angry": [],
                    "Anxious/Scared": [],
                    "Happy/Content": [],
                    "Others, but positive feelings": [],
                    "Others, but negative feelings": []
                },
            },

            "after_classification_negative": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_specific_event(
                    user_id, app, db_session),

                "choices": {
                    "Yes, something happened": "event_is_recent",
                    "No, it's just a general feeling": "more_questions",
                },
                "protocols": {
                    "Yes, something happened": [],
                    "No, it's just a general feeling": []
                },
            },

            "event_is_recent": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_event_is_recent(
                    user_id, app, db_session),

                "choices": {
                    "It was recent": "revisiting_recent_events",
                    "It was distant": "revisiting_distant_events",
                },
                "protocols": {
                    "It was recent": [],
                    "It was distant": []
                },
            },

            "revisiting_recent_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_recent(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "more_questions",
                    "No": "more_questions",
                },
                "protocols": {
                    "Yes": [self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[8]],
                    "No": [self.PROTOCOL_TITLES[11]],
                },
            },

            "revisiting_distant_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_distant(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "more_questions",
                    "No": "more_questions",
                },
                "protocols": {
                    "Yes": [self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[17]],
                    "No": [self.PROTOCOL_TITLES[6]]
                },
            },

            "more_questions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_more_questions(
                    user_id, app, db_session),

                "choices": {
                    "Continue": "check_targetA",
                    "No, I'd like to see suggestions": "suggestions",
                    "No, I'd like to end the session": "ending_prompt",
                },
                "protocols": {
                    "Continue": [],
                    "No, I'd like to see suggestions": [],
                    "No, I'd like to end the session": [],
                },
            },

            "check_targetA": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_targetA(user_id,
                                                                                                            app,
                                                                                                            db_session),
                "choices": {
                    "Yes": "targetA_finder",
                    "No": "internal_persecutor_accusing",
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "targetA_finder": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_targetA_finder(user_id,
                                                                                                             app,
                                                                                                             db_session),
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_targetA_name(user_id)
                },
                "protocols": {"open_text": []},
            },

            "displaying_antisocial_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_antisocial_emotion(
                    user_id, app, db_session),
                # Envy, jealousy, greed, hatred, mistrust, malevolence, or revengefulness
                "choices": {
                    "Anger": lambda user_id, db_session, curr_session, app: self.get_angry_feeling(user_id),
                    "Envy": lambda user_id, db_session, curr_session, app: self.get_envy_feeling(user_id),
                    "Greed": lambda user_id, db_session, curr_session, app: self.get_greed_feeling(user_id),
                    "Hatred": lambda user_id, db_session, curr_session, app: self.get_hatred_feeling(user_id),
                    "Mistrust": lambda user_id, db_session, curr_session, app: self.get_mistrust_feeling(user_id),
                    "Vengefulness": lambda user_id, db_session, curr_session, app: self.get_vengefulness_feeling(
                        user_id),
                    "Others": "specify_antisocial_emotion",
                    "No, I don't have negative emotions towards them": "check_denial",
                },
                "protocols": {
                    "Anger": [],
                    "Envy": [],
                    "Greed": [],
                    "Hatred": [],
                    "Mistrust": [],
                    "Vengefulness": [],
                    "Others": [],
                    "No, I don't have negative emotions towards them": [],
                },
            },

            "specify_antisocial_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_user_feeling(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_user_feeling(user_id)
                },
                "protocols": {"open_text": []},
            },

            "check_denial": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_denial(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "denial",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question_after_denial(
                        user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "denial": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_denial(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": []
                },
            },

            "displacement": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_displacement(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": []
                },
            },

            "check_A_antisocial_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_A_antisocial_emotion(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "check_fight",
                    "No": "displaying_antisocial_behavior",
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "check_fight": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_fight(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "projective_identification",
                    "No": "projection",
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "projective_identification": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_projective_identification(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            "projection": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_projection(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            "displaying_antisocial_behavior": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_antisocial_behavior(
                    user_id, app, db_session),

                "choices": {
                    "Yes": lambda user_id, db_session, curr_session, app: self.get_next_question_after_takeout(user_id),
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "check_targetB": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_targetB(user_id,
                                                                                                            app,
                                                                                                            db_session),
                "choices": {
                    "Yes": "targetB_finder",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "targetB_finder": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_targetB_finder(user_id,
                                                                                                             app,
                                                                                                             db_session),
                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.save_targetB_name(user_id)
                },
                "protocols": {"open_text": []},
            },

            "check_regression": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_regression(user_id,
                                                                                                               app,
                                                                                                               db_session),
                "choices": {
                    "Yes": "regression",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "regression": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_regression(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            "check_transferance": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_transferance(
                    user_id,
                    app,
                    db_session),
                "choices": {
                    "Yes": "transferance",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "transferance": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_transference(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            "check_reaction_formation": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_check_reaction_formation(
                    user_id,
                    app,
                    db_session),
                "choices": {
                    "Yes": "reaction_formation",
                    "No": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "reaction_formation": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_reaction_formation(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },

            "internal_persecutor_accusing": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_accusing(user_id,
                                                                                                              app,
                                                                                                              db_session),

                "choices": {
                    "Yes": "check_projection_internal",
                    "No": "no_mechanism_detected",
                },
                "protocols": {
                    "Yes": [],
                    "No": [],
                },
            },

            "check_projection_internal": {
                "model_prompt": lambda user_id, db_session, curr_session,
                                       app: self.get_model_prompt_check_projection_internal(
                    user_id, app, db_session),

                "choices": {
                    "Yes": "projection_internal",
                    "No": "check_targetA"
                },
                "protocols": {
                    "Yes": [],
                    "No": []
                },
            },

            "projection_internal": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_projection_internal(
                    user_id),

                "choices": {
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "End the session": [],
                },
            },


            "after_classification_positive": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_happy(user_id, app,
                                                                                                           db_session),

                "choices": {
                    "Continue": "check_emotion_after_happy",
                    "See suggestions": "suggestions",
                    "End the session": "ending_prompt"
                },
                "protocols": {
                    "Continue": [],
                    "See suggestions": [],
                    "End the session": []
                },
            },

            "check_emotion_after_happy": {
                "model_prompt": lambda user_id, db_session, curr_session,
                                       app: self.get_model_prompt_check_emotion_after_happy(
                    user_id, app, db_session),

                "choices": {
                    "Sad": lambda user_id, db_session, curr_session, app: self.get_sad_emotion(user_id),
                    "Angry": lambda user_id, db_session, curr_session, app: self.get_angry_emotion(user_id),
                    "Anxious/Scared": lambda user_id, db_session, curr_session, app: self.get_anxious_emotion(user_id),
                    "Other negative feelings": lambda user_id, db_session, curr_session,
                                                      app: self.get_negative_emotion(user_id),

                },
                "protocols": {
                    "Sad": [],
                    "Angry": [],
                    "Anxious/Scared": [],
                    "Other negative feelings": []
                },
            },

            "no_mechanism_detected": {
                "model_prompt": lambda user_id, db_session, curr_session,
                                       app: self.get_model_prompt_no_mechanism_detected(user_id, app, db_session),

                "choices": {
                    "See suggestions": "suggestions",
                    "Replay": "restart_prompt",
                    "End the session": "ending_prompt",
                },
                "protocols": {
                    "See suggestions": [],
                    "Replay": [],
                    "End the session": [],
                },
            },

            "suggestions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggestions(
                    user_id, app, db_session),

                "choices": {
                    "Yes, I'd love to": "trying_protocol",
                    "No, I'd like to end the session": "ending_prompt",
                },
                "protocols": {
                    "Yes, I'd love to": [],
                    "No, I'd like to end the session": [],
                },
            },

            "trying_protocol": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_protocol(
                    user_id, app, db_session),

                "choices": {"Continue": "user_found_useful",
                            "End the session": "ending_prompt",
                            },
                "protocols": {"Continue": [], "End the session": []},
            },

            "tip1": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_tip1(
                    user_id, app, db_session),
                "choices": {"Continue": "tip2",
                            "End": "ending_prompt",
                            },
                "protocols": {"Continue": [],
                              "End": [], },
            },

            "tip2": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_tip2(
                    user_id, app, db_session),
                "choices": {"End": "ending_prompt",
                            },
                "protocols": {"End": [], },
            },

            "user_found_useful": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(
                    user_id, app, db_session),

                "choices": {
                    "I feel better": "new_protocol_better",
                    "I feel worse": "new_protocol_worse",
                    "I feel no change": "new_protocol_same",
                },
                "protocols": {
                    "I feel better": [],
                    "I feel worse": [],
                    "I feel no change": []
                },
            },

            "new_protocol_better": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id,
                                                                                                                app,
                                                                                                                db_session),

                "choices": {
                    "Yes, I'd like to see other suggestions": "tip1",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes, I'd like to see other suggestions": [],
                    "No (end session)": []
                },
            },

            "new_protocol_worse": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id,
                                                                                                               app,
                                                                                                               db_session),

                "choices": {
                    "Yes, I'd like to see other suggestions": "tip1",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes, I'd like to see other suggestions": [],
                    "No (end session)": []
                },
            },

            "new_protocol_same": {
                "model_prompt": [
                    "I am sorry to hear you have not detected any change in your mood.",
                    "That can sometimes happen but if you agree we could try another protocol and see if that is more helpful to you.",
                    "Would you like me to suggest a different protocol?"
                ],

                "choices": {
                    "Yes, I'd like to see other suggestions": "tip1",
                    "No (end session)": "ending_prompt",
                },
                "protocols": {
                    "Yes, I'd like to see other suggestions": [],
                    "No (end session)": []
                },
            },

            "ending_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ending(user_id,
                                                                                                            app,
                                                                                                            db_session),

                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []}
            },

            "restart_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_restart_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening_restart(
                        user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },
        }
        self.QUESTION_KEYS = list(self.QUESTIONS.keys())

    def initialise_prev_questions(self, user_id):
        self.recent_questions[user_id] = []

    def clear_persona(self, user_id):
        self.chosen_personas[user_id] = ""

    def clear_names(self, user_id):
        self.users_names[user_id] = ""

    def clear_datasets(self, user_id):
        self.datasets[user_id] = pd.DataFrame(columns=['sentences'])

    def initialise_remaining_choices(self, user_id):
        self.remaining_choices[user_id] = ["check_reaction_formation", "check_targetB",
                                           "check_transferance", "check_regression"]

    def save_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["ask_name"]
        except:  # noqa
            user_response = ""
        self.users_names[user_id] = user_response
        return "intro_prompt"

    def save_targetA_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["targetA_finder"].lstrip().rstrip()
        except:  # noqa
            user_response = ""
        self.targetA_names[user_id] = user_response if user_response != "" else "X"
        return "displaying_antisocial_emotion"

    def save_targetB_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["targetB_finder"].lstrip().rstrip()
        except:  # noqa
            user_response = ""
        self.targetA_names[user_id] = user_response if user_response != "" else "Y"
        return "displacement"

    def save_user_feeling(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["specify_antisocial_emotion"]
        except:  # noqa
            user_response = "negative feeling"
        self.users_feelings[user_id] = user_response
        return "check_A_antisocial_emotion"

    def clear_suggestions(self, user_id):
        self.suggestions[user_id] = []
        self.reordered_protocol_questions[user_id] = deque(maxlen=5)

    def clear_emotion_scores(self, user_id):
        self.guess_emotion_predictions[user_id] = ""

    def create_new_run(self, user_id, db_session, user_session):
        new_run = UserModelRun(session_id=user_session.id)
        db_session.add(new_run)
        db_session.commit()
        self.current_run_ids[user_id] = new_run.id
        return new_run

    def clear_choices(self, user_id):
        self.user_choices[user_id] = {}

    def get_intro_prompt(self, user_id):
        name = self.users_names[user_id] if self.users_names[user_id] != "" else "there"
        welcome = "Hi " + name + "! Nice to meet you!"
        intro_prompt0 = "My name is Alex, your emotional support assistant."
        intro_prompt1 = "I'm here to help externalize your emotions to your childhood-self and analyze them from a third person point of view."
        intro_prompt2 = "This makes it easier to recognize defense mechanisms that you unconsciously employ when you encounter negative emotions."
        intro_prompt3 = "To start, please could you tell me how you are feeling today?"
        return welcome, intro_prompt0, intro_prompt1, intro_prompt2, intro_prompt3


    def get_restart_prompt(self, user_id):
        name = self.users_names[user_id] if self.users_names[user_id] != "" else "my friend"
        restart_prompt = [
            "Please could you tell me again, " + self.users_names[user_id] + ", how are you feeling today?"]
        return restart_prompt

    def get_next_question(self, user_id):
        if self.remaining_choices[user_id] == []:
            return "no_mechanism_detected"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice

    def get_next_question_after_takeout(self, user_id):
        self.remaining_choices[user_id].remove("check_targetB")  # delete check_targetB from remaining_choices

        if self.remaining_choices[user_id] == []:
            return "no_mechanism_detected"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice

    def get_next_question_after_denial(self, user_id):
        self.remaining_choices[user_id].remove("check_targetB")
        self.remaining_choices[user_id].remove(
            "check_reaction_formation")  # delete check_targetB from remaining_choices

        if self.remaining_choices[user_id] == []:
            return "no_mechanism_detected"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice

    def get_denial(self, user_id):
        question = "Thank you, I really appreciate your cooperation. It looks like you are using denial mechanism. Denial usually happens when you refuse to accept facts or reality, thus block external events from awareness."
        return self.split_sentence(question)

    def get_displacement(self, user_id):
        question = "Thank you, I really appreciate your time and patience. From what you've told me, it is possible that you are using displacement mechanism. This happens when you redirect your negative emotion from its original source to a less threatening recipient."
        question += " In your case, it looks like you redirect your {} from {} to {}.".format(
            self.users_feelings[user_id], self.targetA_names[user_id], self.targetB_names[user_id])
        return self.split_sentence(question)

    def get_transference(self, user_id):
        question = "Great job, you've done so well with this. Thank you very much for your time and efforts. It seems like you are using transference mechanism. This usually happens when you redirect some of your feelings about one person (usually your primary caregiver) to an entirely different person."
        question += " In your case, it looks like you transfer your {} from your primary caregiver to {} due to their similarities.".format(
            self.users_feelings[user_id], self.targetA_names[user_id])
        return self.split_sentence(question)

    def get_regression(self, user_id):
        question = "Thank you very much, I really appreciate your cooperation. I think you are probably using regression mechanism. It is a way to protect yourself in stressful events by going back in time to a period when you safe, instead of handling the unacceptable facts in a more adult manner."
        return self.split_sentence(question)

    def get_projection(self, user_id):
        question = "Great job, I really appreciate your cooperation. From what you've told me, you are probably using projection mechanism. Projection happens when you recognize your unacceptable thoughts in someone else to avoid recognizing those thoughts in yourself subconsciously."
        question += " In your case, it's likely you attribute your {} to {}.".format(
            self.users_feelings[user_id], self.targetA_names[user_id])
        return self.split_sentence(question)

    def get_projection_internal(self, user_id):
        question = "Thank you very much! From what you have told me, it's likely that you are using the internal persecutor. It's possible you direct the internal persecutor to yourself and thus develop an inner persecutor as a way to protect yourself."
        return self.split_sentence(question)

    def get_reaction_formation(self, user_id):
        question = "Thanks a lot, I really appreciate your cooperation. It's possible that you are using reaction formation mechanism. This usually happens when you unconsciously behaves in the opposite way to which you think or feel, in order to hide your true feelings."
        question += " In your case, it looks like you treat {} in a very friendly way to hide your {}.".format(
            self.targetA_names[user_id], self.users_feelings[user_id])
        return self.split_sentence(question)

    def get_projective_identification(self, user_id):
        question = "Well done, thank you for your trust and cooperation! It looks like you are using projective identification mechanism. This usually happens when you continuously project unacceptable thoughts onto another person, and that person ends up acting or feeling in ways that combine both your projection and their feelings."
        question += " In your case, it's likely you attribute your {} to {}, and stubbornly accuse {} of showing this negative feeling towards you. Eventually, {} unconsciously accepts this projection, and behaves like they hold this negative attitude towards you, which may cause a fight as a result.".format(
            self.users_feelings[user_id], self.targetA_names[user_id], self.targetA_names[user_id],
            self.targetA_names[user_id])
        return self.split_sentence(question)

    def determine_next_prompt_opening(self, user_id, app, db_session):
        user_response = self.user_choices[user_id]["choices_made"]["intro_prompt"]
        emotion = get_emotion(user_response)
        if emotion == 'fear':
            self.guess_emotion_predictions[user_id] = 'anxious'
            self.user_emotions[user_id] = 'anxious'
        elif emotion == 'sadness':
            self.guess_emotion_predictions[user_id] = 'sad'
            self.user_emotions[user_id] = 'sad'
        elif emotion == 'anger':
            self.guess_emotion_predictions[user_id] = 'angry'
            self.user_emotions[user_id] = 'angry'
        else:
            self.guess_emotion_predictions[user_id] = 'happy'
            self.user_emotions[user_id] = 'happy'
        return "guess_emotion"

    def determine_next_prompt_opening_restart(self, user_id, app, db_session):
        user_response = self.user_choices[user_id]["choices_made"]["restart_prompt"]
        print(user_response)
        emotion = get_emotion(user_response)
        print(emotion)
        if emotion == 'fear':
            self.guess_emotion_predictions[user_id] = 'anxious'
            self.user_emotions[user_id] = 'anxious'
        elif emotion == 'sadness':
            self.guess_emotion_predictions[user_id] = 'sad'
            self.user_emotions[user_id] = 'sad'
        elif emotion == 'anger':
            self.guess_emotion_predictions[user_id] = 'angry'
            self.user_emotions[user_id] = 'angry'
        else:
            self.guess_emotion_predictions[user_id] = 'happy'
            self.user_emotions[user_id] = 'happy'
        return "guess_emotion"

    def get_best_sentence(self, column, prev_qs):
        maxscore = 0
        chosen = ''
        for row in column.dropna().sample(n=5):  # was 25
            fitscore = get_sentence_score(row, prev_qs)
            if fitscore > maxscore:
                maxscore = fitscore
                chosen = row
        if chosen != '':
            return chosen
        else:
            return random.choice(column.dropna().sample(n=5).to_list())  # was 25

    def split_sentence(self, sentence):
        temp_list = re.split('(?<=[.?!]) +', sentence)
        if '' in temp_list:
            temp_list.remove('')
        temp_list = [i + " " if i[-1] in [".", "?", "!"] else i for i in temp_list]
        print(temp_list)
        if len(temp_list) == 2:
            return temp_list[0], temp_list[1]
        elif len(temp_list) == 3:
            return temp_list[0], temp_list[1], temp_list[2]
        elif len(temp_list) == 4:
            return temp_list[0], temp_list[1], temp_list[2], temp_list[3]
        elif len(temp_list) == 5:
            return temp_list[0], temp_list[1], temp_list[2], temp_list[3], temp_list[4]
        else:
            return sentence

    def get_model_prompt_guess_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["From what you have said I believe you are feeling {}. Is this correct?"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.guess_emotion_predictions[user_id].lower())
        return self.split_sentence(question)

    def get_model_prompt_check_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data[
            "I am sorry. Please select from the emotions below the one that best reflects what you are feeling:"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_model_prompt_check_emotion_after_happy(self, user_id, app, db_session):
        question = "Bravo, thanks for your cooperation. Please select from the emotions below the one that best reflects what you felt:"
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_positive_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "positive"
        self.user_emotions[user_id] = "positive"
        return "after_classification_positive"

    def get_negative_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "negative"
        self.user_emotions[user_id] = "negative"
        return "after_classification_negative"

    def get_angry_feeling(self, user_id):
        self.users_feelings[user_id] = "anger"
        return "check_A_antisocial_emotion"

    def get_envy_feeling(self, user_id):
        self.users_feelings[user_id] = "envy"
        return "check_A_antisocial_emotion"

    def get_greed_feeling(self, user_id):
        self.users_feelings[user_id] = "greed"
        return "check_A_antisocial_emotion"

    def get_hatred_feeling(self, user_id):
        self.users_feelings[user_id] = "hatred"
        return "check_A_antisocial_emotion"

    def get_mistrust_feeling(self, user_id):
        self.users_feelings[user_id] = "mistrust"
        return "check_A_antisocial_emotion"

    def get_vengefulness_feeling(self, user_id):
        self.users_feelings[user_id] = "vengefulness"
        return "check_A_antisocial_emotion"

    def get_user_feeling(self, user_id):
        question = "Alright, thanks. Could you please specify your childhood-self's negative feelings towards " + \
                   self.targetA_names[
                       user_id] + "? If you don't know what specific feeling your childhood-self has, please input 'general negative feeling' or just send me a blank message."
        return self.split_sentence(question)

    def get_sad_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "sad"
        self.user_emotions[user_id] = "sad"
        return "after_classification_negative"

    def get_angry_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "angry"
        self.user_emotions[user_id] = "angry"
        return "after_classification_negative"

    def get_anxious_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "anxious"
        self.user_emotions[user_id] = "anxious"
        return "after_classification_negative"

    def get_happy_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "happy"
        self.user_emotions[user_id] = "happy"
        return "after_classification_positive"

    def get_model_prompt_project_emotion(self, user_id, app, db_session):
        question = "Thank you, I will help you deal with your emotions in a moment. Before I do that, could you please try to project your negatiive feeling onto your childhood self? Take your time to try this, and press 'Continue' when you feel ready. And I will ask you some questions about your childhood-self to get a better understanding of your situation, does it sound good to you?"
        return self.split_sentence(question)

    def get_model_prompt_accusing(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Are you always blaming and accusing yourself for when something goes wrong?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_specific_event(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "specific event"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_event_is_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Was this caused by a recent or distant event (or events)?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_revisit_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Have you recently attempted protocol 9 and found this reignited unmanageable emotions as a result of old events?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_revisit_distant(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Have you recently attempted protocol 10 and found this reignited unmanageable emotions as a result of old events?"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_more_questions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        base_prompt = "Thank you. Now I will ask some questions to understand your situation."
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        question1 = "Before I do that, could you please try to project your negative feeling onto your childhood self?"
        question2 = "Take your time to try this, and press 'Continue' when you feel ready."
        question3 = "Does it sound good to you?"
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question), question1, question2, question3

    def get_model_check_targetA(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["check targeta"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_model_check_targetB(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["check targetb"].dropna()  # to-do delete name request
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.users_feelings[user_id], self.targetA_names[user_id])
        return self.split_sentence(question)

    def get_model_targetA_finder(self, user_id, app, db_session):
        question = "Okay, thank you for your response. Could you please tell me the name of this person if it's convenient? You may send me a blank messgae or input 'X' if you are not comfortable sharing the name, and I fully understand your concern."
        return self.split_sentence(question)

    def get_model_targetB_finder(self, user_id, app, db_session):
        question = "May I have the name of this person please, if you don't mind? You may send me a blank messgae or input 'Y' if you are not comfortable sharing their names, I completely understand your concern."
        return self.split_sentence(question)

    def get_model_check_denial(self, user_id, app, db_session):
        question = "You are doing a great job so far! If it's ok to proceed, may I ask you to think one step further? Would you agree it's possible that you actually have some negative feeling towards " + \
                   self.targetA_names[
                       user_id] + ", but somehow you are not fully aware of it or choose to deny it? I'm sorry that my words may sound a little bit intrusive, but I am trying to understand your emotion better and hope you understand that denial is common and normal, especially in times of great stress."
        return self.split_sentence(question)

    def get_model_prompt_check_projection_internal(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["childhood trauma"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_model_check_regression(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["childlike"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        return self.split_sentence(question)

    def get_model_check_transferance(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["caregiver"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        question += " Especially when " + self.targetA_names[
            user_id] + " and their primary caregiver look alike or have similar personalities."
        return self.split_sentence(question)

    def get_model_check_reaction_formation(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["friendly"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        return self.split_sentence(question)

    def get_model_antisocial_emotion(self, user_id, app, db_session):
        question = "Thank you. We all feel emotions and sometimes express them towards other people, and sometimes these emotions can be quite strong. Could you tell me whether your childhood-self has been feeling any of the following negative emotions towards " + \
                   self.targetA_names[
                       user_id] + "?"
        return self.split_sentence(question)

    def get_model_A_antisocial_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["TargetA attitude"].dropna()
        my_string = self.get_best_sentence(column, prev_qs) + " Particularly, " + self.users_feelings[
            user_id] + ", like your childhood-self has towards " + self.targetA_names[user_id] + " ?"
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        return self.split_sentence(question)

    def get_model_check_fight(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["fight"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.targetA_names[user_id])
        return self.split_sentence(question)

    def get_model_antisocial_behavior(self, user_id, app, db_session):
        question = "I see, thank you for taking the time to answer all these questions, I am still trying to figure out the best way to help. I wonder if your childhood-self has openly expressed their " + \
                   self.users_feelings[user_id] + " towards " + self.targetA_names[
                       user_id] + "?"
        return self.split_sentence(question)

    def get_model_prompt_no_mechanism_detected(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["Fail to detect"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_happy(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["Recall painful event"].dropna()
        question = self.get_best_sentence(column,
                                          prev_qs) + " Please press 'Continue' when you are ready. Or if you are not in the mood, I also have some suggestions on how to handle difficult emotions if you are interested by any chance. Feel free to end the conversation whenever you feel uncomfortable."
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_suggestions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["Here are my tips"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_trying_protocol(self, user_id, app, db_session):
        question = "First of all, it is essential to recognize your negative emotions and deal with them in a mature way. Actually, there is a defense mechanism called 'sublimation', which means to displace your unacceptable emotions into constructive and socially acceptable behaviors. Maybe you can try to practice protocol 8, 17 or 18,  I really hope the protocols will make you feel better. Please press 'Continue' when you finish."
        return self.split_sentence(question)

    def get_model_prompt_tip1(self, user_id, app, db_session):
        question = "Adopting a thir-person perspective is helpful to reduce the intensity of your negative emotions. Hence, why not try protocol 15 and 19 when you get stuck in negative emotion. Please press 'Continue' if you would like to know more."
        return self.split_sentence(question)

    def get_model_prompt_tip2(self, user_id, app, db_session):
        question = "Ultimately, laughter is the best medicine. Digesting negative emotions into laughter will make you feel better. Try protocol 11, 12, 13, 14 and 16, and learn to laugh at your failures or disturbing events."
        return self.split_sentence(question)

    def get_model_prompt_found_useful(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["All emotions - Do you feel better or worse after having taken this protocol?"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_new_better(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["Patient feels better"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_new_worse(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id], columns=['sentences'])
        data = self.datasets
        column = data["Patient feels worse"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_ending(self, user_id, app, db_session):
        question = "Thank you for taking part. I appreciate that you've spent this time with me. I look forward to seeing you next time. You have been disconnected. Refresh the page if you would like to start over."
        return self.split_sentence(question)

    def update_conversation(self, user_id, new_dialogue, db_session, app):
        try:
            session_id = self.user_choices[user_id]["current_session_id"]
            curr_session = UserModelSession.query.filter_by(id=session_id).first()
            if curr_session.conversation is None:
                curr_session.conversation = "" + new_dialogue
            else:
                curr_session.conversation = curr_session.conversation + new_dialogue
            curr_session.last_updated = datetime.datetime.utcnow()
            db_session.commit()
        except KeyError:
            curr_session = UserModelSession(
                user_id=user_id,
                conversation=new_dialogue,
                last_updated=datetime.datetime.utcnow(),
            )

            db_session.add(curr_session)
            db_session.commit()
            self.user_choices[user_id]["current_session_id"] = curr_session.id

    def save_current_choice(
            self, user_id, input_type, user_choice, user_session, db_session, app
    ):
        # Set up dictionary if not set up already
        # with Session() as session:

        try:
            self.user_choices[user_id]
        except KeyError:
            self.user_choices[user_id] = {}

        # Define default choice if not already set
        try:
            current_choice = self.user_choices[user_id]["choices_made"][
                "current_choice"
            ]
        except KeyError:
            current_choice = self.QUESTION_KEYS[0]

        try:
            self.user_choices[user_id]["choices_made"]
        except KeyError:
            self.user_choices[user_id]["choices_made"] = {}

        if current_choice == "ask_name":
            self.clear_suggestions(user_id)
            self.user_choices[user_id]["choices_made"] = {}
            self.create_new_run(user_id, db_session, user_session)

        # Save current choice
        self.user_choices[user_id]["choices_made"]["current_choice"] = current_choice
        self.user_choices[user_id]["choices_made"][current_choice] = user_choice

        curr_prompt = self.QUESTIONS[current_choice]["model_prompt"]
        # prompt_to_use = curr_prompt
        if callable(curr_prompt):
            curr_prompt = curr_prompt(user_id, db_session, user_session, app)

        # removed stuff here

        else:
            self.update_conversation(
                user_id,
                "Model:{} \nUser:{} \n".format(curr_prompt, user_choice),
                db_session,
                app,
            )

        if current_choice == "guess_emotion":
            option_chosen = user_choice + " ({})".format(
                self.guess_emotion_predictions[user_id]
            )
        else:
            option_chosen = user_choice
        choice_made = Choice(
            choice_desc=current_choice,
            option_chosen=option_chosen,
            user_id=user_id,
            session_id=user_session.id,
            run_id=self.current_run_ids[user_id],
        )
        db_session.add(choice_made)
        db_session.commit()

        return choice_made

    def determine_next_choice(
            self, user_id, input_type, user_choice, db_session, user_session, app
    ):
        # Find relevant user info by using user_id as key in dict.
        #
        # Then using the current choice and user input, we determine what the next
        # choice is and return this as the output.

        # Some edge cases to consider based on the different types of each field:
        # May need to return list of model responses. For next protocol, may need
        # to call function if callable.

        # If we cannot find the specific choice (or if None etc.) can set user_choice
        # to "any".

        # PRE: Will be defined by save_current_choice if it did not already exist.
        # (so cannot be None)

        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        current_choice_for_question = self.QUESTIONS[current_choice]["choices"]
        current_protocols = self.QUESTIONS[current_choice]["protocols"]
        if input_type != "open_text":
            if current_choice == "check_emotion":
                if user_choice == "Sad":
                    next_choice = current_choice_for_question["Sad"]
                    protocols_chosen = current_protocols["Sad"]
                elif user_choice == "Angry":
                    next_choice = current_choice_for_question["Angry"]
                    protocols_chosen = current_protocols["Angry"]
                elif user_choice == "Anxious/Scared":
                    next_choice = current_choice_for_question["Anxious/Scared"]
                    protocols_chosen = current_protocols["Anxious/Scared"]
                elif user_choice == "Others, but positive feelings":
                    next_choice = current_choice_for_question["Others, but positive feelings"]
                    protocols_chosen = current_protocols["Others, but positive feelings"]
                elif user_choice == "Others, but negative feelings":
                    next_choice = current_choice_for_question["Others, but negative feelings"]
                    protocols_chosen = current_protocols["Others, but negative feelings"]
                else:
                    next_choice = current_choice_for_question["Happy/Content"]
                    protocols_chosen = current_protocols["Happy/Content"]
            else:
                print(user_choice)
                next_choice = current_choice_for_question[user_choice]
                protocols_chosen = current_protocols[user_choice]

        else:
            next_choice = current_choice_for_question["open_text"]
            protocols_chosen = current_protocols["open_text"]

        if callable(next_choice):
            next_choice = next_choice(user_id, db_session, user_session, app)

        if current_choice == "guess_emotion" and user_choice == "Yes":
            if self.guess_emotion_predictions[user_id] == "sad":
                next_choice = next_choice["Sad"]
            elif self.guess_emotion_predictions[user_id] == "angry":
                next_choice = next_choice["Angry"]
            elif self.guess_emotion_predictions[user_id] == "anxious":
                next_choice = next_choice["Anxious/Scared"]
            else:
                next_choice = next_choice["Happy/Content"]

        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(user_id, db_session, user_session, app)
        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]
        if callable(next_prompt):
            next_prompt = next_prompt(user_id, db_session, user_session, app)

        # Case: new suggestions being created after first protocol attempted
        if next_choice == "opening_prompt":
            self.clear_suggestions(user_id)
            self.clear_emotion_scores(user_id)
            self.create_new_run(user_id, db_session, user_session)

        # else:
        next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())
        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice
        return {"model_prompt": next_prompt, "choices": next_choices}
