import time
import torch

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


sentences = {
    "en": [
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
    ],
    "fr": [
        "C'est le meilleur jour de ma vie !",
        "J'adore absolument ce produit, il fonctionne parfaitement.",
        "Le temps est magnifique aujourd'hui, si rafraîchissant.",
        "J'ai passé un moment incroyable à la fête hier soir.",
        "Ce film est fantastique, je le recommande vivement.",
        "Je suis tellement reconnaissant pour tout le soutien que j'ai reçu.",
        "Le service dans ce restaurant est exceptionnel.",
        "Je me sens si heureux et content en ce moment.",
        "Ce livre est incroyablement inspirant et bien écrit.",
        "Je viens d'être promu au travail, je suis ravi !",
        "Le coucher de soleil aujourd'hui était à couper le souffle.",
        "Je suis si fier de mon équipe pour leur travail acharné.",
        "C'est le repas le plus délicieux que j'aie jamais mangé.",
        "Je me sens si chanceux d'avoir des amis aussi merveilleux.",
        "Le concert d'hier soir était absolument phénoménal.",
        "Je suis fou de joie face aux résultats de mes efforts.",
        "Ces vacances ont été un rêve devenu réalité.",
        "Je suis tellement excité pour l'événement à venir !",
        "Le service client ici est de premier ordre.",
        "Je me sens tellement motivé et prêt à relever mes objectifs.",
        "C'est la meilleure nouvelle que j'ai entendue toute la semaine !",
        "Je suis tellement impressionné par la qualité de ce produit.",
        "L'ambiance dans ce café est si chaleureuse et accueillante.",
        "Je suis tellement reconnaissant pour cette incroyable opportunité.",
        "Cette chanson me met toujours de bonne humeur.",
        "Je me sens tellement énergisé et vivant aujourd'hui.",
        "La présentation s'est mieux passée que prévu !",
        "Je suis si heureux de la façon dont tout s'est déroulé.",
        "C'est le plus amusant que j'ai eu depuis longtemps.",
        "Je suis tellement excité de voir ce que l'avenir nous réserve.",
        "Le travail d'équipe aujourd'hui était absolument impeccable.",
        "Je me sens tellement béni d'être entouré d'autant d'amour.",
        "C'est la façon parfaite de terminer la journée.",
        "Je suis si content d'avoir pris la décision de venir ici.",
        "La nouvelle mise à jour de l'application est incroyablement utile.",
        "Je me sens tellement inspiré après cette conversation.",
        "C'est le meilleur cadeau que j'aie jamais reçu.",
        "Je suis si fier de moi pour avoir atteint cet objectif.",
        "La vue d'ici est absolument magnifique.",
        "Je me sens si détendu et en paix en ce moment.",
        "C'est exactement ce dont j'avais besoin d'entendre aujourd'hui.",
        "Je suis tellement excité de commencer ce nouveau chapitre.",
        "La nourriture ici est toujours délicieuse et fraîche.",
        "Je me sens tellement confiant et prêt pour le défi.",
        "C'est le geste le plus attentionné qui soit.",
        "Je suis si heureux de faire partie de cette équipe incroyable.",
        "L'événement a été un énorme succès, bravo !",
        "Je me sens tellement optimiste pour l'avenir.",
        "C'est la meilleure décision que j'aie jamais prise.",
        "C'est le pire jour que j'aie jamais vécu.",
        "Je ne supporte pas ce produit, il est complètement inutile.",
        "Le temps est affreux aujourd'hui, si déprimant.",
        "J'ai passé un moment terrible à la fête hier soir.",
        "Ce film est horrible, je ne le recommanderais à personne.",
        "Je suis tellement frustré par tous les problèmes auxquels j'ai été confronté.",
        "Le service dans ce restaurant est terrible.",
        "Je me sens si triste et déçu en ce moment.",
        "Ce livre est ennuyeux et mal écrit.",
        "Je viens d'être rétrogradé au travail, je suis dévasté.",
        "Le coucher de soleil aujourd'hui était terne et sans intérêt.",
        "Je suis tellement déçu par mon équipe pour leur manque d'effort.",
        "C'est le repas le plus dégoûtant que j'aie jamais mangé.",
        "Je me sens si malchanceux d'avoir des amis aussi toxiques.",
        "Le concert d'hier soir a été un désastre complet.",
        "Je suis tellement contrarié par les résultats de mes efforts.",
        "Ces vacances ont été un cauchemar.",
        "Je suis tellement anxieux à propos de l'événement à venir.",
        "Le service client ici est affreux.",
        "Je me sens tellement démotivé et coincé dans une routine.",
        "C'est la pire nouvelle que j'ai entendue toute la semaine.",
        "Je suis tellement déçu par la qualité de ce produit.",
        "L'ambiance dans ce café est si froide et inhospitalière.",
        "Je suis tellement frustré par cette opportunité manquée.",
        "Cette chanson me met toujours de mauvaise humeur.",
        "Je me sens tellement vidé et épuisé aujourd'hui.",
        "La présentation a horriblement mal tourné !",
        "Je suis si mécontent de la façon dont tout s'est déroulé.",
        "C'est le moment le plus ennuyeux que j'ai eu depuis longtemps.",
        "Je suis tellement inquiet de ce que l'avenir nous réserve.",
        "Le travail d'équipe aujourd'hui a été un échec complet.",
        "Je me sens si mal aimé et isolé en ce moment.",
        "C'est la pire façon de terminer la journée.",
        "Je regrette tellement la décision de venir ici.",
        "La nouvelle mise à jour de l'application est complètement inutile.",
        "Je me sens tellement découragé après cette conversation.",
        "C'est le pire cadeau que j'aie jamais reçu.",
        "Je suis tellement déçu de moi pour ne pas avoir atteint cet objectif.",
        "La vue d'ici est absolument terrible.",
        "Je me sens tellement stressé et anxieux en ce moment.",
        "C'est la dernière chose dont j'avais besoin d'entendre aujourd'hui.",
        "Je suis tellement nerveux à l'idée de commencer ce nouveau chapitre.",
        "La nourriture ici est toujours fade et rassis.",
        "Je me sens tellement peu sûr de moi et mal préparé pour le défi.",
        "C'est le geste le plus irréfléchi qui soit.",
        "Je suis si malheureux de faire partie de cette équipe toxique.",
        "L'événement a été un échec complet, quel désastre.",
        "Je me sens tellement pessimiste pour l'avenir.",
        "C'est la pire décision que j'aie jamais prise."
    ]
}

# Enable TensorFloat-32 for Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")

# 0 for GPU, -1 for CPU
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline("sentiment-analysis", device=device, model_kwargs={"torch_dtype": torch.float16})

# Time the classification
start_time = time.time()
results = classifier(sentences["en"])
end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / len(sentences["en"])

print(f"\nUsing {'GPU' if device == 0 else 'CPU'}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per sentence: {avg_time:.3f} seconds")

# for i, result in enumerate(results):
#     print(f"label: {result['label']} with score: {round(result['score'], 4)}, sentence: {sentences['en'][i]}")

# Use different model for French language
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", device=device, model=model, tokenizer=tokenizer)

# Time the classification
start_time = time.time()
results = classifier(sentences["fr"])
end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / len(sentences["fr"])

print(f"\nUsing {'GPU' if device == 0 else 'CPU'}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per sentence: {avg_time:.3f} seconds")

# for i, result in enumerate(results):
#     print(f"label: {result['label']}, score: {round(result['score'], 4)}, sentence: {sentences['fr'][i]}")

