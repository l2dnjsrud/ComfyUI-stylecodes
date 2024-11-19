Weights: https://huggingface.co/CiaraRowles/stylecodes

Paper: WIP

Project Page: https://ciarastrawberry.github.io/stylecodes.github.io/


Recent advances in Image generation have propelled capabilities to new heights, however existing methods of control for these models are insufficient for many use cases and social contexts.

One of the most impressive recent examples of both beneficial controllability and the packaging of that in a way that allows for social control of generation models are the proliferation of srefs, these are codes used with MidJourneys image generation tool to encode a specific image style into a short numeric code. Allowing users to quickly share with their friends how to make images in a similar style to their art.

The only main drawback with this tool currently is both the inability to generate your own code from an image and the lack of an knowledge on how to reproduce this functionality. Stylecodes attempts to resolve both these issues by providing an open research solution to generating codes from images with a custom style Encoder.

<img width="2122" alt="Demo-Image_3" src="https://github.com/user-attachments/assets/e7150b5e-8ea6-46ae-afa0-b8e48b8104d6">


How to use

```
pip install -r requirements.txt

download the models from here: [[https://huggingface.co/CiaraRowles/IP-Adapter-Instruct](https://huggingface.co/CiaraRowles/stylecodes)](https://huggingface.co/CiaraRowles/stylecodes)

place them in the "models" folder

run either demo.py, demo_make_stylecode.py or demo_use_stylecode.py with the provided args (demo just works as normal)

```

Notes:
The current implimentation is overfit to the dataset, fixable later
The source dataset isn't trained on digital art or cartoon or anime or realism, don't expect those to work very well.
