from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      cache_dir='./cache')

from PruneTransformers import BertForSequenceClassification as QBert
quantModel = QBert.from_pretrained('bert-base-uncased', cache_dir='./cache')