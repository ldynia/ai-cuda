import time
import torch

from transformers import pipeline


sentences = [
    "This is the best day of my life!",
    "I absolutely love this product, it works perfectly.",
    "The weather is beautiful today, so refreshing.",
    "I had an amazing time at the party last night.",
    "This movie is fantastic, I highly recommend it.",
    "I'm so grateful for all the support I've received.",
    "The service at this restaurant is outstanding.",
    "I feel so happy and content right now.",
    "This book is incredibly inspiring and well-written.",
    "I just got promoted at work, I'm thrilled!",
    "The sunset today was breathtaking.",
    "I'm so proud of my team for their hard work.",
    "This is the most delicious meal I've ever had.",
    "I feel so lucky to have such wonderful friends.",
    "The concert last night was absolutely phenomenal.",
    "I'm overjoyed with the results of my efforts.",
    "This vacation has been a dream come true.",
    "I'm so excited for the upcoming event!",
    "The customer service here is top-notch.",
    "I feel so motivated and ready to tackle my goals.",
    "This is the best news I've heard all week!",
    "I'm so impressed with the quality of this product.",
    "The atmosphere in this café is so cozy and welcoming.",
    "I'm so thankful for this incredible opportunity.",
    "This song always puts me in a great mood.",
    "I feel so energized and alive today.",
    "The presentation went better than I expected!",
    "I'm so happy with how everything turned out.",
    "This is the most fun I've had in a long time.",
    "I'm so excited to see what the future holds.",
    "The teamwork today was absolutely flawless.",
    "I feel so blessed to be surrounded by such love.",
    "This is the perfect way to end the day.",
    "I'm so glad I made the decision to come here.",
    "The new update to the app is incredibly useful.",
    "I feel so inspired after that conversation.",
    "This is the best gift I've ever received.",
    "I'm so proud of myself for achieving this goal.",
    "The view from here is absolutely stunning.",
    "I feel so relaxed and at peace right now.",
    "This is exactly what I needed to hear today.",
    "I'm so excited to start this new chapter.",
    "The food here is always delicious and fresh.",
    "I feel so confident and ready for the challenge.",
    "This is the most thoughtful gesture ever.",
    "I'm so happy to be part of this amazing team.",
    "The event was a huge success, well done!",
    "I feel so optimistic about the future.",
    "This is the best decision I've ever made.",
    "This is the worst day I've ever had.",
    "I can't stand this product, it's completely useless.",
    "The weather is awful today, so depressing.",
    "I had a terrible time at the party last night.",
    "This movie is horrible, I wouldn't recommend it to anyone.",
    "I'm so frustrated with all the problems I've been facing.",
    "The service at this restaurant is terrible.",
    "I feel so sad and disappointed right now.",
    "This book is boring and poorly written.",
    "I just got demoted at work, I'm devastated.",
    "The sunset today was dull and unremarkable.",
    "I'm so disappointed in my team for their lack of effort.",
    "This is the most disgusting meal I've ever had.",
    "I feel so unlucky to have such toxic friends.",
    "The concert last night was a complete disaster.",
    "I'm so upset with the results of my efforts.",
    "This vacation has been a nightmare.",
    "I'm so anxious about the upcoming event.",
    "The customer service here is awful.",
    "I feel so unmotivated and stuck in a rut.",
    "This is the worst news I've heard all week.",
    "I'm so unimpressed with the quality of this product.",
    "The atmosphere in this café is so cold and unwelcoming.",
    "I'm so frustrated with this missed opportunity.",
    "This song always puts me in a bad mood.",
    "I feel so drained and exhausted today.",
    "The presentation went horribly wrong!",
    "I'm so unhappy with how everything turned out.",
    "This is the most boring time I've had in a long time.",
    "I'm so worried about what the future holds.",
    "The teamwork today was a complete failure.",
    "I feel so unloved and isolated right now.",
    "This is the worst way to end the day.",
    "I'm so regretful about the decision to come here.",
    "The new update to the app is completely useless.",
    "I feel so discouraged after that conversation.",
    "This is the worst gift I've ever received.",
    "I'm so disappointed in myself for not achieving this goal.",
    "The view from here is absolutely terrible.",
    "I feel so stressed and anxious right now.",
    "This is the last thing I needed to hear today.",
    "I'm so nervous about starting this new chapter.",
    "The food here is always bland and stale.",
    "I feel so insecure and unprepared for the challenge.",
    "This is the most thoughtless gesture ever.",
    "I'm so unhappy to be part of this toxic team.",
    "The event was a complete failure, what a disaster.",
    "I feel so pessimistic about the future.",
    "This is the worst decision I've ever made."
]

# Enable TensorFloat-32 for Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")

# 0 for GPU, -1 for CPU
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("sentiment-analysis", device=device, model_kwargs={"torch_dtype": torch.float32})

# Time the classification
start_time = time.time()
results = classifier(sentences * 100)
end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / len(sentences)

print(f"\nUsing {'GPU' if device == 0 else 'CPU'}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per sentence: {avg_time:.3f} seconds")

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
