import random
import string
import itertools

class Wordle:
    def __init__(self):
        self.max_guesses = 10
        self.dictionary = self.generate_possible_words()
    
    def generate_possible_words(self):
        return [''.join(word) for word in itertools.product(string.ascii_lowercase, repeat=4)]
    
    def get_feedback(self, guess, actual):
        feedback = []
        for i in range(4):
            if guess[i] == actual[i]:
                feedback.append('=')
            elif guess[i] in actual:
                feedback.append('+')
            else:
                feedback.append('-')
        return ''.join(feedback)

    def filter_words(self, dictionary, guess, feedback):
        filtered_words = []
        for word in dictionary:
            if self.get_feedback(guess, word) == feedback:
                filtered_words.append(word)
        return filtered_words
    
    def make_guess(self, dictionary):
        return random.choice(dictionary)
    
    def backtrack(self, dictionary, guesses, actual_word):
        if guesses == self.max_guesses:
            return False
        if not dictionary:
            return False

        guess = self.make_guess(dictionary)
        feedback = self.get_feedback(guess, actual_word)
        print(f"Guess #{guesses + 1}: {guess}, Feedback: {feedback}")

        if feedback == "====":
            print(f"The word has been guessed correctly: {guess}")
            return True

        new_dictionary = self.filter_words(dictionary, guess, feedback)
        return self.backtrack(new_dictionary, guesses + 1, actual_word)

    def play_game(self):
        print("Think of a 4-letter word.")
        actual_word = input("Enter the actual 4-letter word: ").strip()
        if len(actual_word) != 4 or not all(c in string.ascii_lowercase for c in actual_word):
            print("Invalid word. Please enter a valid 4-letter word.")
            return

        if self.backtrack(self.dictionary, 0, actual_word):
            print("The program guessed the word successfully.")
        else:
            print("The program couldn't guess the word within 10 guesses.")

if __name__ == "__main__":
    game = Wordle()
    game.play_game()
