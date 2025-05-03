from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
# classifier = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english") #GIVES POSITIVE AND NEGATIVE
desc = {
    "Guardians of the Galaxy" : "A group of intergalactic criminals are forced to work together to stop a fanatical warrior from taking control of the universe." , 
    "The Avengers" : "Earth's mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army from enslaving humanity." ,
    "The Dark Knight" : "When the menace known as the Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham. The Dark Knight must accept one of the greatest psychological and physical tests of his ability to fight injustice." ,
    "The Godfather" : "An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son." ,
    "The lost city of Z" : "A true-life drama, centering on British explorer Percy Fawcett, who disappeared in the Amazon in 1925 while searching for a fabled civilization." ,
}

ans = {}
for i in range(len(desc)):
    ans[list(desc.keys())[i]] = classifier(list(desc.values())[i])
    print(list(desc.keys())[i], " : ", list(desc.values())[i])
    print("Emotion : ", ans[list(desc.keys())[i]][0]['label'])
    print("Score : ", ans[list(desc.keys())[i]][0]['score'])
    print()

