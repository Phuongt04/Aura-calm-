SPEECH EMOTION RECOGNITION EVALUATION
=====================================

Architecture:
CNN-GRU window encoder + Temporal BiLSTM

Probability vector:
p_s = [psA, psH, psN, psS]

Class order:
[A, H, N, S] =
[angry, happy, neutral, sad]

Probability constraint:
psA + psH + psN + psS = 1

Predicted label:
y_hat_s = argmax(p_s)

Reliability/confidence score:
q_s = max(p_s)

Test samples:
30

Accuracy:
0.6667

Macro-F1:
0.6382

Weighted-F1:
0.6517

Mean confidence q_s:
0.7007

Mean q_s for correct predictions:
0.7804

Mean q_s for incorrect predictions:
0.5414