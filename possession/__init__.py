"""Frame-level ball possession for the More than a Hack 2027 challenge.

    data.py      load the tactical ATD, homography, Smart Tagging and public GT (cached)
    signals.py   per-frame physical signals derived from the tracking
    evidence.py  per-frame log-likelihoods for the H / A / D states
    decode.py    offline Viterbi over H / A / D
    pipeline.py  data in, labels out -- the one function make_submission.py calls
"""
