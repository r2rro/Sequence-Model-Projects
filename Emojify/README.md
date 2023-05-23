## Emojifier - LSTM in Keras
The emojifier code will use word vector representations to make your messages more expressive.

Rather than writing:
>"Congratulations on the promotion! Let's get coffee and talk. Love you!"   

The emojifier can automatically turn this into:
>"Congratulations on the promotion! 👍  Let's get coffee and talk. ☕️ Love you! ❤️"

We will implement a model which inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (⚾️).

## Model Overview
Here is the Emojifier model overview implemented:

<p align="center">
  <img width="500" src="https://github.com/r2rro/Sequence-Model-Projects/blob/main/Emojify/images/emojifier-v2.png" alt="lstm">
</p>

