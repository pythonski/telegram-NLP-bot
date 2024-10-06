import dspy


class GenerateAnswer(dspy.Signature):
    """
    Answer questions with short factoid answers.
    If no answer can be found based on the context, return: I'm sorry, I cannot help you with this.
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="keep it short and simple: often between 1 and 5 words")

class GenerateQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class MakeAnswerFriendly(dspy.Signature):
    """
    Given an input short answer, write it as a more complete sentence as the answer to the question.
    It should still be kept short and sweet.
    """

    question = dspy.InputField(desc="question that was answered")
    original_answer = dspy.InputField(desc="contains relevant information")
    helpful_answer = dspy.OutputField()

class DetermineInputType(dspy.Signature):
    """
    Given a text, determine if it is:
    - a question about the project (return question)
    - an entry to the project diary (return entry)
    - something else (return something)
    Questions typically end with a question mark ?, but not necessarily.
    """

    text = dspy.InputField()
    input_type = dspy.OutputField(desc="question, entry, or something")

class GenerateEntrySummary(dspy.Signature):
    """
    Generate a shortish overview of a diary entry, keeping all the relevant information (dates, people, named entities) but getting rid of
    unnecessary fluff. 
    """

    entry = dspy.InputField()
    summary = dspy.OutputField(desc="Only the summary itself, without reasoning or other entries")
