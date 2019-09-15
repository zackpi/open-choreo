# open-choreo
Teaches you how to dance like Bruno Mars and other famous dancers!

Open-choreo uses openpose to see well you match your favorite choreography video. It compares your pose against those in dance videos and refuses to let you go on until you've mastered your move.

Open-chorea uses advanced computer vision and machine learning modules such as OpenCV, Tensorflow, and websockets! But don't worry about that... just dance ♪┏(・o･)┛♪┗ ( ･o･) ┓♪


### Server

The server calculates poses using a state-of-the-art-server with a high-end GPU called "turtle" - thanks `alamp`.

To run this server, type:
`python posenet-python server.py`

This server takes in base64 encoded images, calculates pose information using posenet, and returns the coordinates of your nose, mouth, and other bodily organs in medium-dimensional matrix.

### Webcam

The client of the server is the webcam. The webcam takes each frame, converts it to a base64 string, and sends it to the server.

Then it processes the returned string and does a complex smoothing algorithm* that we invented ourselves (!!!). This makes sures that the feedback you're seeing of your moves are TOP NOTCH.

* the smoothing algorithm takes historical frames and figures out the best result for the move.

### Scoring

We also invented this algorithm, yipee!!! But yeah, it just tells you how good you are lol - calculates the error between your pose and the dance video's pose. So get good already geez.