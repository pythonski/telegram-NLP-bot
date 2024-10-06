import dspy
from dsp.utils import deduplicate
from retriever import DSPythonicRMClient
from dspy_signatures import GenerateAnswer, GenerateQuery, DetermineInputType, MakeAnswerFriendly, GenerateEntrySummary


class QA_RAG(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateQuery) for _ in range(max_hops)]
        self.retrieve = DSPythonicRMClient()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


qa_pipeline = QA_RAG()
classify_type = dspy.ChainOfThought(DetermineInputType)
helper = dspy.ChainOfThought(MakeAnswerFriendly)
summarize = dspy.ChainOfThought(GenerateEntrySummary)