from transformers import pipeline


QA_MODEL = "deepset/roberta-base-squad2"
QG_MODEL = "gpssohi/distilbart-qgen-3-3"

qa_pipeline = pipeline('question-answering',
                       model=QA_MODEL,
                       tokenizer=QA_MODEL)
qg_pipeline = pipeline('text2text-generation',
                       model=QG_MODEL,
                       tokenizer=QG_MODEL)


input_text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
q = qg_pipeline(input_text)
print(q)

qtext = q[0]['generated_text']
score = qa_pipeline(context=input_text, question=qtext)
print(score)