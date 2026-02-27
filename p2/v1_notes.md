See paper2.md
https://docs.google.com/document/d/1jkWllMTo13DKOXWgWj_XopQ8ad-1giuDg9WZAjjD1ew/edit?pli=1&tab=t.0

Suggested quick collection checklist (minimal but strong)

For each person (S01, S02, S03):
	•	handshake: 20 clips (mix close/normal/far)
	•	none: 20 clips (hand present sometimes + no hand sometimes)
	•	wave: 10 clips
	•	highfive: 10 clips

```
python data_collect.py --subject S01 --env indoor_bright --out ./dataset --fps 30 --duration 2.5
```

Are we doing multi gesture system paper?

No — you are NOT doing multi-gesture identification in this paper.

Wave / high-five clips are:
	•	Rejection stress tests
	•	Specificity validation
	•	Safety validation

Keep scope tight.

⸻

Now go collect data with that in mind:

Binary task.
Confusers for robustness.
Not multi-class recognition.
