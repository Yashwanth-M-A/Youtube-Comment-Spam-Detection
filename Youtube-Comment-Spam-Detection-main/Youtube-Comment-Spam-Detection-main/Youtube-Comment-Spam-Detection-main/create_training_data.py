import pandas as pd

data = {
    "CONTENT": [
        "Subscribe now for free gifts!",
        "Great video, very informative!",
        "Click this link to win a prize!",
        "Thanks for the helpful tutorial.",
        "Limited time offer, buy now!",
        "Nice work, I learned a lot.",
        "Earn money quickly with this method!",
        "Awesome content, keep it up!",
        "Visit our website for discounts!",
        "Really enjoyed this video, thanks!",
        "Buy cheap followers here!",
        "Excellent explanation and well presented.",
        "Get your free trial today!",
        "Super helpful tips, thank you!",
        "Act fast, sale ends soon!",
        "Brilliant tutorial, very clear.",
        "Claim your cash prize now!",
        "This is the best video on this topic.",
        "Win big with our exclusive deal!",
        "Happy to have found this, great job!",
        "Join now and save big!",
        "Perfect content, very useful.",
        "Don't miss this limited time offer!",
        "Fantastic insights, well done!",
        "Risk-free investment, sign up!",
        "Excellent content, thanks for sharing.",
        "Get rich quick, no effort needed!",
        "Very clear and easy to understand.",
        "Subscribe now, don't miss out!",
        "Good job, learned a lot from this."
    ],
    "LABEL": [
        "spam", "not_spam", "spam", "not_spam", "spam", "not_spam", "spam", "not_spam", "spam", "not_spam",
        "spam", "not_spam", "spam", "not_spam", "spam", "not_spam", "spam", "not_spam", "spam", "not_spam",
        "spam", "not_spam", "spam", "not_spam", "spam", "not_spam", "spam", "not_spam", "spam", "not_spam"
    ]
}

df = pd.DataFrame(data)
df.to_csv("labeled_comments.csv", index=False)
print("labeled_comments.csv file created successfully!")
