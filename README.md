

```python
#Jericho Herrera



#Observations:
    
    #Observation 1: CNN has a negative, and also the lowest, compound sentiment score. 
     #   It could mean that their tweets have to do with slightly more negative news compared to 
    #    the other media outlets, however, being that it is actually very close to 0, 
    #    (CNN's score being -0.009598 to be exact), 
    #    they could actually just have the most neutral sentiments.
    
    #Observation 2: CBS has the highest compound sentiment score. 
    #    Their news could have more positive sentiments than the other media outlets.
    
    #Observation 3: CNN has never had a tweet (over the past 100 tweets) that has had 
    #    a negative compound sentiment score below -0.75.
```


```python
import json
import tweepy
import apikeys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
consumer_key = apikeys.TWITTER_CONSUMER_KEY
consumer_secret = apikeys.TWITTER_CONSUMER_SECRET
access_token = apikeys.TWITTER_ACCESS_TOKEN
access_token_secret = apikeys.TWITTER_ACCESS_TOKEN_SECRET

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
bbc = "@bbc"
cbs = "@cbs"
cnn = "@cnn"
fox = "@FoxNews"
nyt = "@nytimes"
```


```python
#BBC 


bbc_text_list = []
bbc_date_list = []
bbc_negative_list = []
bbc_neutral_list = []
bbc_positive_list = []
bbc_compound = []


bbc_public_tweets = api.user_timeline(bbc, count = 100)

for bbc_tweet in bbc_public_tweets:
    
    bbc_text = bbc_tweet['text']
    bbc_date = bbc_tweet['created_at']
    
    print(bbc_text)
    
    
    bbc_scores = analyzer.polarity_scores(bbc_text)
    print(bbc_scores)
    print(bbc_date)
    print(' ')
    
    bbc_text_list.append(bbc_text)
    bbc_date_list.append(bbc_date)
    bbc_negative_list.append(bbc_scores['neg'])
    bbc_neutral_list.append(bbc_scores['neu'])
    bbc_positive_list.append(bbc_scores['pos'])
    bbc_compound.append(bbc_scores['compound'])
    

```

    RT @BBCOne: SO. MUCH. CUTE. 😍
    #AttenboroughandtheGiantElephant https://t.co/4UyVvh6qBm
    {'neg': 0.0, 'neu': 0.56, 'pos': 0.44, 'compound': 0.6915}
    Sun Dec 10 21:34:02 +0000 2017
     
    RT @BBCEarth: 'Never before have we had such an awareness of what we are doing to the planet and never before have we had the power to do s…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 21:02:27 +0000 2017
     
    🌹@DuaLipa performing 'Homesick' was a completely stunning way to end this series of #SLFN. 💖
    Watch in full 👉… https://t.co/LgMlMfnv7F
    {'neg': 0.0, 'neu': 0.855, 'pos': 0.145, 'compound': 0.4391}
    Sun Dec 10 21:00:06 +0000 2017
     
    RT @BBCEarth: 'What shocks me ...is how fast things are changing here. We’re headed into uncharted territory' - @expeditionlog
    #BluePlanet2…
    {'neg': 0.126, 'neu': 0.874, 'pos': 0.0, 'compound': -0.3818}
    Sun Dec 10 20:58:15 +0000 2017
     
    RT @BBCOne: If we don’t act, coral reefs could be gone by the end of this century. 😢 #BluePlanet2 https://t.co/OdOSKBtz0S
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:57:07 +0000 2017
     
    RT @BBCEarth: In 1986, many nations decided to end commercial whaling, and fishermen are now reporting pods of whales with numbers never se…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:49:42 +0000 2017
     
    Little things can make a big difference.
    #BluePlanet2 via @BBCEarth https://t.co/hyzwA06cNw
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:45:04 +0000 2017
     
    RT @BBCSpringwatch: Manmade materials are destroying #OurBluePlanet - this is how
    #BluePlanet2 
    https://t.co/sEccQzjv1r https://t.co/7ZmDOz…
    {'neg': 0.231, 'neu': 0.769, 'pos': 0.0, 'compound': -0.5574}
    Sun Dec 10 20:43:59 +0000 2017
     
    RT @BBCEarth: It’s estimated that tens of millions of sharks are killed each year from getting trapped in fishing nets
    #BluePlanet2 https:/…
    {'neg': 0.283, 'neu': 0.717, 'pos': 0.0, 'compound': -0.836}
    Sun Dec 10 20:43:18 +0000 2017
     
    RT @BBCOne: These are *all* items regurgitated by birds. 😥 #BluePlanet2 https://t.co/WyFtZHjKjx
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:24:26 +0000 2017
     
    RT @BBCOne: Our rubbish, our responsibility. ✊ #BluePlanet2 
    
    Find out more about what you can do to help: https://t.co/LiA3x5I2gk. https:/…
    {'neg': 0.0, 'neu': 0.87, 'pos': 0.13, 'compound': 0.4019}
    Sun Dec 10 20:24:19 +0000 2017
     
    Eight delightful dolphin photos that'll inspire you to help clean up our oceans. #BluePlanet2 🐬❤️️… https://t.co/PP5l7QrqFP
    {'neg': 0.0, 'neu': 0.482, 'pos': 0.518, 'compound': 0.9169}
    Sun Dec 10 20:20:04 +0000 2017
     
    RT @BBCEarth: ‘We’re only now beginning to realise what an impact noise is having on the inhabitants of the ocean’ – David Attenborough
    #Bl…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:14:12 +0000 2017
     
    RT @BBCOne: "The oceans are under threat now as never ever before in human history." - Sir David Attenborough. #BluePlanet2 https://t.co/yn…
    {'neg': 0.152, 'neu': 0.848, 'pos': 0.0, 'compound': -0.5267}
    Sun Dec 10 20:04:45 +0000 2017
     
    RT @BBCEarth: The countdown is on! The final episode of #BluePlanet2: Our Blue Planet starts 8pm GMT on @BBCOne, BBC Earth Nordics &amp; @BBCEa…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 19:58:55 +0000 2017
     
    RT @BBCSpringwatch: Seven British conservation heroes you should know about - this lot deserve endless 👏🏽👏🏽
    #BluePlanet2 #OurBluePlanet 
    ht…
    {'neg': 0.0, 'neu': 0.837, 'pos': 0.163, 'compound': 0.5106}
    Sun Dec 10 19:58:19 +0000 2017
     
    'Good luck, little leatherback' - Sir David Attenborough 🐢
    
    #BluePlanet2 | 8PM | @BBCOne https://t.co/vMliWBZFBE
    {'neg': 0.0, 'neu': 0.604, 'pos': 0.396, 'compound': 0.7096}
    Sun Dec 10 19:50:01 +0000 2017
     
    'There's a huge bomb up there. I'm 76% sure. It's not like the penguin hostage situation.' 🐧😂
    #ScotSquad via… https://t.co/x8I92iMImk
    {'neg': 0.213, 'neu': 0.602, 'pos': 0.185, 'compound': -0.1803}
    Sun Dec 10 19:30:02 +0000 2017
     
    Miles of cable and thousands of twinkling bulbs have been used to create the longest festive tunnel of light in Eur… https://t.co/C22wpljo2u
    {'neg': 0.0, 'neu': 0.797, 'pos': 0.203, 'compound': 0.6249}
    Sun Dec 10 19:00:02 +0000 2017
     
    Tonight, David Attenborough investigates the remarkable story of Jumbo the elephant. Attenborough and the Giant Ele… https://t.co/4NK1B5ZNwr
    {'neg': 0.0, 'neu': 0.816, 'pos': 0.184, 'compound': 0.5574}
    Sun Dec 10 18:33:05 +0000 2017
     
    98-year-old Mary wins Christmas, and our hearts. 🎁❤️️
    Via @BBCLookNorth. https://t.co/4gOl16THuM
    {'neg': 0.0, 'neu': 0.73, 'pos': 0.27, 'compound': 0.5719}
    Sun Dec 10 18:00:06 +0000 2017
     
    RT @BBCTwo: Who wants a Sneaky Peak at the next episode of #PeakyBlinders? 😱 https://t.co/tzCB83ceSS
    {'neg': 0.137, 'neu': 0.863, 'pos': 0.0, 'compound': -0.2263}
    Sun Dec 10 17:51:08 +0000 2017
     
    RT @bbcstories: "I think people my age think that death is something to be scared of but I think it's like being free and at peace" - Meet…
    {'neg': 0.116, 'neu': 0.555, 'pos': 0.329, 'compound': 0.8765}
    Sun Dec 10 17:50:55 +0000 2017
     
    📻 How to deal with mental health issues in the family. https://t.co/48BOLabkXK https://t.co/7Eh3EjYywg
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 17:30:06 +0000 2017
     
    This 12-year-old has created a cheap way to test lead levels in water using a mobile phone app. 🚰📱Via @BBCClick. https://t.co/TWqttOHJX3
    {'neg': 0.0, 'neu': 0.9, 'pos': 0.1, 'compound': 0.25}
    Sun Dec 10 17:00:01 +0000 2017
     
    📞👽 We've been listening for signs of life from other worlds for decades with little success. But what should we do… https://t.co/e3IETfS8cs
    {'neg': 0.0, 'neu': 0.86, 'pos': 0.14, 'compound': 0.5279}
    Sun Dec 10 16:30:07 +0000 2017
     
    Thousands of internet users have come together to buy this stunning castle... although it does need a little TLC. 🏰… https://t.co/qhkNd4HyM5
    {'neg': 0.0, 'neu': 0.864, 'pos': 0.136, 'compound': 0.4588}
    Sun Dec 10 16:00:06 +0000 2017
     
    That was unexpected. 😂
    
    Michael McIntyre's Big Show via @BBCOne. https://t.co/2QUZXppIbQ
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 15:30:04 +0000 2017
     
    #SkiSunday is back! ⛷🎿 Join the team in Val D'Isere for the men's slalom. 5:15pm on @BBCTwo. https://t.co/mPdHEuxHOX https://t.co/w3pqO1lCn6
    {'neg': 0.0, 'neu': 0.878, 'pos': 0.122, 'compound': 0.3595}
    Sun Dec 10 15:03:04 +0000 2017
     
    💰 @DanTDM has been named the highest-earning YouTuber of 2017 - making £12.3m this year, according to @Forbes magaz… https://t.co/yVEcwct3Es
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 14:30:07 +0000 2017
     
    Rebel Wilson reenacts her Pitch Perfect audition and it’s slap-tastic.
    
    The Graham Norton Show | @BBCOne https://t.co/TYszXcf9ZH
    {'neg': 0.083, 'neu': 0.725, 'pos': 0.192, 'compound': 0.4767}
    Sun Dec 10 14:00:06 +0000 2017
     
    The BBC has made the 1930s issues of the complete Radio Times magazines available online for the first time. 📻 Chec… https://t.co/aeK8TTggDW
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 13:30:05 +0000 2017
     
    From Dark, to the return of The Crown &amp; a #DoctorWho Christmas special, here are the TV shows worth watching in Dec… https://t.co/4qq2MlpONj
    {'neg': 0.0, 'neu': 0.813, 'pos': 0.187, 'compound': 0.5574}
    Sun Dec 10 13:00:07 +0000 2017
     
    RT @bbcthree: In 2011 BBC Three made a documentary about a boy who defied his bullies. 
    
    Now #EverybodysTalkingAboutJamie again. https://t.…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 12:43:52 +0000 2017
     
    'There's ignorance about employment law &amp; people don't want to hear complaints.'
    Classical music has a sexual haras… https://t.co/PpdlbV0Gck
    {'neg': 0.189, 'neu': 0.811, 'pos': 0.0, 'compound': -0.4063}
    Sun Dec 10 12:30:03 +0000 2017
     
    'Their birthdays are never easy.' 
    Hannah is one of thousands of mums living with PTSD from experiencing traumatic… https://t.co/lPMW5VXHfa
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 12:00:03 +0000 2017
     
    How do you make puppets fall in love? ❤️️🎎
    Puppeteers reveal their techniques. https://t.co/GPNqLrFZit https://t.co/HfB09Yw8ek
    {'neg': 0.0, 'neu': 0.769, 'pos': 0.231, 'compound': 0.6369}
    Sun Dec 10 11:30:06 +0000 2017
     
    RT @CBeebiesHQ: He's the disco dancing, geography guide with the moves! 😎
    
    @BBCStrictly doesn't know what's hit it! 😆
    
    #GoJetters🕺
    
    #Strict…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 11:15:39 +0000 2017
     
    RT @BBCBreakfast: A clairvoyant cat? 🐈 Or a turtle in the know?🐢
    Animals in Russia are predicting the outcome of @FIFAWorldCup 2018 https:/…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 11:05:15 +0000 2017
     
    RT @BBCOne: David Attenborough uncovers the story of Jumbo – a celebrity animal superstar who is said to have inspired the movie Dumbo. Ton…
    {'neg': 0.0, 'neu': 0.868, 'pos': 0.132, 'compound': 0.4939}
    Sun Dec 10 11:04:13 +0000 2017
     
    Watch @ScottishBallet's incredible interpretations of Stravinsky. https://t.co/NGZuhnKutW https://t.co/vUOkoaJ7mc
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 11:00:06 +0000 2017
     
    ✊ @Time magazine has named 'the Silence Breakers' - women who spoke out against sexual abuse &amp; harassment - as its… https://t.co/roSq8US4Ev
    {'neg': 0.312, 'neu': 0.688, 'pos': 0.0, 'compound': -0.8271}
    Sun Dec 10 10:30:05 +0000 2017
     
    RT @BBCWthrWatchers: Some great pics of #UKsnow sent in by the early risers! https://t.co/b9Of6bvvIz
    {'neg': 0.0, 'neu': 0.747, 'pos': 0.253, 'compound': 0.6588}
    Sun Dec 10 10:23:23 +0000 2017
     
    RT @BBCR1: 🎄 BIG FESTIVE NEWS 🎄
    
    We've invited some of your favourite stars to take over Radio 1 on Christmas Day!
    
    Get ready for Superstar…
    {'neg': 0.0, 'neu': 0.763, 'pos': 0.237, 'compound': 0.7597}
    Sun Dec 10 10:21:03 +0000 2017
     
    RT @BBCEarth: These orca just want a meal, but they could be risking their lives. 💙#BluePlanet2 https://t.co/JeLlETfohs
    {'neg': 0.163, 'neu': 0.773, 'pos': 0.064, 'compound': -0.4215}
    Sun Dec 10 10:19:47 +0000 2017
     
    'Whilst I appreciate the lovely people slowing down, if you see somebody with a guide dog, just drive on. Ziggy's g… https://t.co/CeH3p8geLq
    {'neg': 0.0, 'neu': 0.735, 'pos': 0.265, 'compound': 0.7579}
    Sun Dec 10 10:00:03 +0000 2017
     
    ‘Why won’t he let me leave?!'
    The #BluePlanet2 team had a rather unexpected visitor while filming in the Galapagos.… https://t.co/vvBS7WorXO
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 09:30:03 +0000 2017
     
    The Greek concept of philoxenia, which translates as ‘love of strangers’, is a warmth that makes foreigners feel im… https://t.co/O3BzC7lUp3
    {'neg': 0.0, 'neu': 0.857, 'pos': 0.143, 'compound': 0.4588}
    Sun Dec 10 09:00:08 +0000 2017
     
    A study of nearly 12,000 children in the UK found 25% were overweight or obese at age seven, rising to 35% at 11.… https://t.co/41qA60xnUd
    {'neg': 0.102, 'neu': 0.898, 'pos': 0.0, 'compound': -0.3612}
    Sun Dec 10 08:00:06 +0000 2017
     
    GET EXCITED!! 🎤❤️️🎶
    @Camila_Cabello has set the release date of her debut album Camila for 12 January.… https://t.co/gRtJVD7FEh
    {'neg': 0.0, 'neu': 0.821, 'pos': 0.179, 'compound': 0.5743}
    Sat Dec 09 20:30:12 +0000 2017
     
    Have we got five eight-year-olds robbing houses? No! We've got two bams with triplets. 😂
    #ScotSquad via… https://t.co/HUsw0CjaoY
    {'neg': 0.135, 'neu': 0.865, 'pos': 0.0, 'compound': -0.3595}
    Sat Dec 09 20:00:03 +0000 2017
     
    The history of condoms stretches back around 3,000 years – but they still offer the best protection against STIs.… https://t.co/shw3Jn5og1
    {'neg': 0.0, 'neu': 0.756, 'pos': 0.244, 'compound': 0.7783}
    Sat Dec 09 19:30:08 +0000 2017
     
    Sexually frivolous and morally ambiguous. 
    Meet pro wrestling’s pansexual phenomenon...🏳️‍🌈🤼‍♀️
    Via @BBC5Live. https://t.co/2Asruxhnv8
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 19:00:02 +0000 2017
     
    RT @BBCWthrWatchers: Ok, so #snow can be a pain ...but wow 😍 
    We're all feeling just a little more festive after your great efforts today!…
    {'neg': 0.096, 'neu': 0.468, 'pos': 0.436, 'compound': 0.8911}
    Sat Dec 09 18:39:17 +0000 2017
     
    RT @bbcthree: YOU'VE BEEN LIVING A LIE.
    https://t.co/YUCx48Mv0o
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 18:39:10 +0000 2017
     
    RT @bbcstories: Do our young men lack positive role models? Davis thought so, so he founded The Manhood Academy. #EverydayHeroes https://t.…
    {'neg': 0.092, 'neu': 0.763, 'pos': 0.145, 'compound': 0.3182}
    Sat Dec 09 18:35:51 +0000 2017
     
    RT @BBCEarth: In Cape Verde, @ProjetoBioSal is on a mission to save baby turtles from human impact. 🐢
    #OurBluePlanet w/ @AluciaProd https:/…
    {'neg': 0.0, 'neu': 0.856, 'pos': 0.144, 'compound': 0.4939}
    Sat Dec 09 18:33:32 +0000 2017
     
    RT @BBCSpringwatch: Brrr, a chilly one tonight!
    Luckily we've some gorgeous grey seals in our advent calendar to warm to cockles of your he…
    {'neg': 0.0, 'neu': 0.64, 'pos': 0.36, 'compound': 0.8655}
    Sat Dec 09 18:33:25 +0000 2017
     
    RT @bbcmusic: ✨@RagNBoneManUK brought all the feels to #SLFN with his stunning performance of 'As You Are' 
    Watch in full 👉https://t.co/JmZ…
    {'neg': 0.0, 'neu': 0.885, 'pos': 0.115, 'compound': 0.3818}
    Sat Dec 09 18:33:22 +0000 2017
     
    30 years of Fairytale of New York: 10 true tales behind the @PoguesOfficial's Christmas favourite. 🎁🎄… https://t.co/LClO0krtv0
    {'neg': 0.0, 'neu': 0.851, 'pos': 0.149, 'compound': 0.4215}
    Sat Dec 09 18:30:39 +0000 2017
     
    Ever wondered what London looked like 2000 years ago? 🇬🇧
    #DiggingForBritain, now on @BBCiPlayer 👉… https://t.co/fd74UnWzF4
    {'neg': 0.0, 'neu': 0.857, 'pos': 0.143, 'compound': 0.3612}
    Sat Dec 09 18:00:04 +0000 2017
     
    Tonight, @McInTweet is joined by @JasonManford and @TheVampsband. 🌟🎶😂 Michael McIntyre's Big Show. 8:20pm on… https://t.co/NW4Y5gYger
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 17:34:07 +0000 2017
     
    'I know it's not real because I'm surrounded by people eating crisps.' 😂🎭
    #LiveAtTheApollo via @BBCTwo. https://t.co/1zwUd8oCvi
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 17:00:06 +0000 2017
     
    Alice is a 26-year-old mum. She's married, she has schizophrenia &amp; she is fed up of being portrayed as 'dangerous'… https://t.co/hy2APiuz5L
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 16:30:06 +0000 2017
     
    Astronomers have discovered the most distant 'supermassive' black hole known to science. ☄🔭 https://t.co/OYWmKfE4t4 https://t.co/10PPSeYkT2
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 16:00:06 +0000 2017
     
    Claire Foy talks about what will be her last outing as Britain’s longest serving monarch - and hints at choppy wate… https://t.co/z6WuYfwM3d
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 15:30:07 +0000 2017
     
    I'm celebrating natural beauty - 'Instagram isn't real life.' @IAmSonnyTurner
    These five body positive champions ar… https://t.co/uUr8AHsfgq
    {'neg': 0.0, 'neu': 0.393, 'pos': 0.607, 'compound': 0.9517}
    Sat Dec 09 15:00:08 +0000 2017
     
    Could you live with your partner alone in a remote lighthouse for 6 months?
    Via @BBCWorldService. https://t.co/7gNRygiGKr
    {'neg': 0.125, 'neu': 0.875, 'pos': 0.0, 'compound': -0.25}
    Sat Dec 09 14:30:02 +0000 2017
     
    🎵 Jack Black’s Jumanji theme tune is the best. 😂
    Via @BBCOne. https://t.co/bqpYxQ8cot
    {'neg': 0.0, 'neu': 0.704, 'pos': 0.296, 'compound': 0.6369}
    Sat Dec 09 14:10:03 +0000 2017
     
    Dyslexie is a font that aims to overcome some of the problems that people with dyslexia can have when reading.… https://t.co/d3GZEMdZgc
    {'neg': 0.124, 'neu': 0.876, 'pos': 0.0, 'compound': -0.4019}
    Sat Dec 09 13:35:04 +0000 2017
     
    🔬🌡 Heard of Terrific Scientific, the BBC’s science campaign for primary pupils? 
    
    If you know a brilliant teacher,… https://t.co/DtpAloo5Fe
    {'neg': 0.0, 'neu': 0.699, 'pos': 0.301, 'compound': 0.7845}
    Sat Dec 09 13:00:06 +0000 2017
     
    Many coastal communities are already battling the effects of sea level rise - and it's made worse by the fact that… https://t.co/WRpTIDB2e9
    {'neg': 0.215, 'neu': 0.785, 'pos': 0.0, 'compound': -0.6369}
    Sat Dec 09 12:31:04 +0000 2017
     
    Danielle is breaking deaf world records left right &amp; centre. 🏊❤️️
    Via @BBCTheSocial. https://t.co/XRlJU7UdU7
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 12:00:02 +0000 2017
     
    Do you feel pressure when giving Christmas gifts? @DrRadhaModgil discusses the psychology of giving &amp; receiving gif… https://t.co/DtkRGlOvdW
    {'neg': 0.1, 'neu': 0.682, 'pos': 0.218, 'compound': 0.3818}
    Sat Dec 09 11:30:05 +0000 2017
     
    RT @bbcthree: Lynsey is saving condemned dogs and finding them forever homes. https://t.co/aF7j3gaZil
    {'neg': 0.195, 'neu': 0.805, 'pos': 0.0, 'compound': -0.4404}
    Sat Dec 09 10:59:05 +0000 2017
     
    RT @bbcmusic: Nearly seven months later, we look back on the #OneLoveManchester concert staged by @ArianaGrande after the devastating attac…
    {'neg': 0.185, 'neu': 0.815, 'pos': 0.0, 'compound': -0.6486}
    Sat Dec 09 10:46:31 +0000 2017
     
    RT @BBCr4today: If you do one thing today...
    
    Listen to Gaelynn Lea play the violin #r4today https://t.co/nXixd8e8rO
    {'neg': 0.0, 'neu': 0.87, 'pos': 0.13, 'compound': 0.34}
    Sat Dec 09 10:44:54 +0000 2017
     
    👂🏻📖 @SJ_Watson was an audiology specialist before penning the international bestseller Before I Go To Sleep. 
    5 peo… https://t.co/RqDVSZ8Ntl
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 10:30:05 +0000 2017
     
    Mason has used his pocket money to fund random acts of kindness this year in memory of his brother, Ross. ❤️️ Via… https://t.co/asKJ9me2MY
    {'neg': 0.0, 'neu': 0.88, 'pos': 0.12, 'compound': 0.4588}
    Sat Dec 09 10:00:03 +0000 2017
     
    Who even needs a calculator anyway? 
    Via BBC Minute https://t.co/5LIDp3hZXh
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 09:30:03 +0000 2017
     
    📚 From essays about Cuba’s transformation to a history of US nuclear policy, here are 10 books to read in December.… https://t.co/XvF7FFPWF8
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sat Dec 09 09:00:09 +0000 2017
     
    There's around 500k pieces of space junk orbiting Earth, posing a huge threat to astronauts. The RemoveDebris space… https://t.co/MLFaaVAxta
    {'neg': 0.157, 'neu': 0.737, 'pos': 0.106, 'compound': -0.2732}
    Sat Dec 09 08:00:03 +0000 2017
     
    As @Lord_Sugar's search for his next business partner continues, take an in-depth look at the remaining candidates.… https://t.co/jkK7AHquzr
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 08 19:30:05 +0000 2017
     
    'For me, drag is a device to entertain people and make them laugh.' 💋💅
    Via @BBCTheSocial. https://t.co/urnzq6NJJA
    {'neg': 0.104, 'neu': 0.769, 'pos': 0.126, 'compound': 0.1027}
    Fri Dec 08 19:00:06 +0000 2017
     
    You might think the hipster beard is a new trend, but the mid-19th century was a true golden age for massive facial… https://t.co/dsJojiqMgH
    {'neg': 0.0, 'neu': 0.844, 'pos': 0.156, 'compound': 0.5719}
    Fri Dec 08 18:30:15 +0000 2017
     
    RT @bbcmusic: WHAT A LINE UP 😍
    Tonight's #SLFN guest host: the incredible @DuaLipa 🌹
    Performances from @KasabianHQ, @RagNBoneManUK, @RAYE a…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 08 18:29:57 +0000 2017
     
    RT @bbc5live: 10-year-old William thought he'd lost his camera when it was swept out to sea 📷 🌊
    
    But he's been reunited with it after it wa…
    {'neg': 0.087, 'neu': 0.913, 'pos': 0.0, 'compound': -0.3182}
    Fri Dec 08 18:28:12 +0000 2017
     
    RT @BBCSpringwatch: We've some gorgeous gannets in today's advent calendar. 🎄🎅🏽
    They're busy building nests - I wonder if they can make wre…
    {'neg': 0.0, 'neu': 0.833, 'pos': 0.167, 'compound': 0.6124}
    Fri Dec 08 18:27:20 +0000 2017
     
    RT @BBCOne: ✨ @Chris_Stark reveals the magic behind *that* BBC One Christmas tearjerker. https://t.co/FbLj5SrI6y
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 08 18:26:46 +0000 2017
     
    🗓 🎶 @ClaudiaWinkle and @Claraamfo look back at this year's best music. The Year in Music 2017. 9pm on @BBCTwo.… https://t.co/fqm4A0Wv3C
    {'neg': 0.0, 'neu': 0.811, 'pos': 0.189, 'compound': 0.6369}
    Fri Dec 08 18:03:03 +0000 2017
     
    🎬 From the new Star Wars to the story of the worst movie ever made, here are 9 movies to watch in December. 👉… https://t.co/Zihk2ad0dz
    {'neg': 0.268, 'neu': 0.732, 'pos': 0.0, 'compound': -0.8271}
    Fri Dec 08 17:30:07 +0000 2017
     
    Sharks, bear markets, narwhals &amp; yak shaving - do you know the 
    business beast ABC?
    Full guide here 👉… https://t.co/kgxpnSZf4A
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 08 17:00:05 +0000 2017
     
    Remember how we actually did alright at Eurovision this year? 🎤🎶
    
    10 surprising music moments from 2017 you've alre… https://t.co/hS72dmiWNE
    {'neg': 0.0, 'neu': 0.814, 'pos': 0.186, 'compound': 0.4767}
    Fri Dec 08 16:30:06 +0000 2017
     
    👦🤔👧 Are you smarter than an 11-year-old? Take this quiz &amp; find out... https://t.co/JNbY5UN9QB https://t.co/nZWffIrmGv
    {'neg': 0.0, 'neu': 0.824, 'pos': 0.176, 'compound': 0.4588}
    Fri Dec 08 14:00:08 +0000 2017
     
    RT @bbcmusic: 💖@Harry_Styles is nominated for BBC Music Artist Of The Year, and we STILL have goosebumps from his beautiful 'Girl Crush' pe…
    {'neg': 0.06, 'neu': 0.792, 'pos': 0.147, 'compound': 0.5106}
    Fri Dec 08 13:39:18 +0000 2017
     
    'If you don't get the angle right, it's not going to work.'
    Can you #DoTheHuw? 😂 https://t.co/hYxrN6C790
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 08 13:30:04 +0000 2017
     
    Stewards working at Prince Harry &amp; Meghan Markle's Royal Wedding have been told to watch Suits before the big day.… https://t.co/ErN62dBvF6
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 08 13:00:06 +0000 2017
     
    The perfect warm drink for cold December evenings. ☕️😋 
    Via @BBCFood. https://t.co/clDVQqgSOj
    {'neg': 0.0, 'neu': 0.641, 'pos': 0.359, 'compound': 0.6808}
    Fri Dec 08 12:30:03 +0000 2017
     
    RT @bbcworldservice: "Falls reduced by 60%." Why some care home residents are 'pimping' their walking frames. Follow the link for more from…
    {'neg': 0.0, 'neu': 0.868, 'pos': 0.132, 'compound': 0.4939}
    Fri Dec 08 12:19:35 +0000 2017
     
    Professional clowns must choose a unique facial makeup design – and they have an unusual way of ‘protecting’ it fro… https://t.co/zmkytd7z8v
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 08 12:00:07 +0000 2017
     



```python
bbc_complete_df = pd.DataFrame({'Date (BBC)': bbc_date_list, 'Tweet (BBC)': bbc_text_list, 
                                'Negative Score (BBC)': bbc_negative_list, 'Neutral Score (BBC)': bbc_neutral_list, 
                                'Positive Score (BBC)': bbc_positive_list, 'Compound Score (BBC)': bbc_compound})
bbc_final_df = bbc_complete_df[['Date (BBC)', 'Tweet (BBC)', 'Negative Score (BBC)', 'Neutral Score (BBC)', 
                 'Positive Score (BBC)', 'Compound Score (BBC)']]
bbc_final_df.head()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date (BBC)</th>
      <th>Tweet (BBC)</th>
      <th>Negative Score (BBC)</th>
      <th>Neutral Score (BBC)</th>
      <th>Positive Score (BBC)</th>
      <th>Compound Score (BBC)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sun Dec 10 21:34:02 +0000 2017</td>
      <td>RT @BBCOne: SO. MUCH. CUTE. 😍\n#Attenboroughan...</td>
      <td>0.000</td>
      <td>0.560</td>
      <td>0.440</td>
      <td>0.6915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun Dec 10 21:02:27 +0000 2017</td>
      <td>RT @BBCEarth: 'Never before have we had such a...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sun Dec 10 21:00:06 +0000 2017</td>
      <td>🌹@DuaLipa performing 'Homesick' was a complete...</td>
      <td>0.000</td>
      <td>0.855</td>
      <td>0.145</td>
      <td>0.4391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sun Dec 10 20:58:15 +0000 2017</td>
      <td>RT @BBCEarth: 'What shocks me ...is how fast t...</td>
      <td>0.126</td>
      <td>0.874</td>
      <td>0.000</td>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun Dec 10 20:57:07 +0000 2017</td>
      <td>RT @BBCOne: If we don’t act, coral reefs could...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
bbc_compound_df = pd.DataFrame(bbc_compound)
bbc_compound_df.columns = ['BBC Compound Score']
bbc_compound_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BBC Compound Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.6915</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.4391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.3818</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#CBS

cbs_text_list = []
cbs_date_list = []
cbs_negative_list = []
cbs_neutral_list = []
cbs_positive_list = []
cbs_compound = []


cbs_public_tweets = api.user_timeline(cbs, count = 100)

for cbs_tweet in cbs_public_tweets:
    
    cbs_text = cbs_tweet['text']
    cbs_date = cbs_tweet['created_at']
    
    print(cbs_text)
    
    
    cbs_scores = analyzer.polarity_scores(cbs_text)
    print(cbs_scores)
    print(cbs_date)
    print(' ')
    
    cbs_text_list.append(cbs_text)
    cbs_date_list.append(cbs_date)
    cbs_negative_list.append(cbs_scores['neg'])
    cbs_neutral_list.append(cbs_scores['neu'])
    cbs_positive_list.append(cbs_scores['pos'])
    cbs_compound.append(cbs_scores['compound'])
```

    Due to NFL overrun, CBS is delayed 8 mins in the following ET &amp; CT markets: Hunstville, AL, Bowling Green, KY, MS,… https://t.co/gbwhgooEQd
    {'neg': 0.083, 'neu': 0.917, 'pos': 0.0, 'compound': -0.2263}
    Mon Dec 11 00:26:50 +0000 2017
     
    Due to NFL overrun CBS is delayed 7 mins in the following ET &amp; CT markets Tampa, Chicago, Maryland, Michigan, Wash… https://t.co/driVgCiX7D
    {'neg': 0.087, 'neu': 0.913, 'pos': 0.0, 'compound': -0.2263}
    Mon Dec 11 00:25:54 +0000 2017
     
    RT @NoActivityCBS: If you want the intel, you get the tickles. 😂 Stream the first 5 episodes of #NoActivity now: https://t.co/Wzq9bVOhGN ht…
    {'neg': 0.0, 'neu': 0.936, 'pos': 0.064, 'compound': 0.0772}
    Sun Dec 10 22:49:02 +0000 2017
     
    RT @startrekcbs: .@albinokid and @wcruz73 are breaking new ground in #StarTrekDiscovery. 🏳️‍🌈 Stream chapter 1 on CBS All Access: https://t…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:48:46 +0000 2017
     
    Don’t miss America’s Game! Stream the Army-Navy game LIVE today at 3PM ET with a FREE trial of #CBSAllAccess:… https://t.co/38udnQj2vs
    {'neg': 0.07, 'neu': 0.742, 'pos': 0.189, 'compound': 0.5754}
    Sat Dec 09 18:24:37 +0000 2017
     
    There's still time before we have to say so long! Catch up with Carol Burnett, her original castmates and special g… https://t.co/WTiK8dzaXO
    {'neg': 0.0, 'neu': 0.791, 'pos': 0.209, 'compound': 0.6476}
    Sat Dec 09 01:11:51 +0000 2017
     
    RT @bigbangtheory: It's the biggest best friend breakup of the year... 😱 Stream the latest full episode of #BigBangTheory: https://t.co/iIw…
    {'neg': 0.0, 'neu': 0.697, 'pos': 0.303, 'compound': 0.8126}
    Fri Dec 08 23:35:41 +0000 2017
     
    RT @AmazingRaceCBS: Presenting... the 30th cast of the #AmazingRace! Made up of champions and winners, this is quite literally the most com…
    {'neg': 0.0, 'neu': 0.746, 'pos': 0.254, 'compound': 0.7777}
    Thu Dec 07 18:23:03 +0000 2017
     
    Do you know all the nominees for Song Of The Year? Get familiar with this year's GRAMMY® Award nominees ahead of Mu… https://t.co/I6hhOtfQjb
    {'neg': 0.0, 'neu': 0.863, 'pos': 0.137, 'compound': 0.5423}
    Thu Dec 07 12:07:06 +0000 2017
     
    A new Twilight Zone original series is coming exclusively to CBS All Access in association with @JordanPeele,… https://t.co/fVYnhUASee
    {'neg': 0.0, 'neu': 0.874, 'pos': 0.126, 'compound': 0.3182}
    Wed Dec 06 17:32:28 +0000 2017
     
    RT @SuperiorDonuts: You'll want to check this box! ☑︎ Stream the latest episode of #SuperiorDonuts: https://t.co/h1xhKLCAPl https://t.co/d8…
    {'neg': 0.0, 'neu': 0.91, 'pos': 0.09, 'compound': 0.1511}
    Tue Dec 05 19:21:31 +0000 2017
     
    RT @YoungSheldon: Wondering what it's like behind the scenes of #YoungSheldon? Let star @OfficialRaeganR take you on a tour of the set! htt…
    {'neg': 0.0, 'neu': 0.883, 'pos': 0.117, 'compound': 0.4199}
    Tue Dec 05 19:14:00 +0000 2017
     
    The Kennedy Center Honors recognizes the lifetime contributions of all types of performance artists. Watch the 40th… https://t.co/azF3rW2ptw
    {'neg': 0.0, 'neu': 0.837, 'pos': 0.163, 'compound': 0.5106}
    Tue Dec 05 00:54:16 +0000 2017
     
    RT @YoungSheldon: There's something @elonmusk never told you about @spacex's Falcon 9. 📓  #YoungSheldon https://t.co/bu3W0IC8zN
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Tue Dec 05 00:03:06 +0000 2017
     
    Just like old times, Carol Burnett bumps up the lights and answers questions from the audience. Catch up on The Car… https://t.co/9r7WI8d1tZ
    {'neg': 0.0, 'neu': 0.894, 'pos': 0.106, 'compound': 0.3612}
    Mon Dec 04 20:52:31 +0000 2017
     
    Star comedian @JimCarrey gets emotional when he expresses his heartfelt thanks to Carol Burnett for her positive in… https://t.co/5QCejBcg5T
    {'neg': 0.0, 'neu': 0.496, 'pos': 0.504, 'compound': 0.9217}
    Mon Dec 04 19:22:07 +0000 2017
     
    Take a trip down memory lane with Carol Burnett for a special tribute you'll never forget. Catch up on The Carol Bu… https://t.co/XBINDav614
    {'neg': 0.0, 'neu': 0.813, 'pos': 0.187, 'compound': 0.5213}
    Mon Dec 04 07:14:06 +0000 2017
     
    RT @JLawlorNY: So, ok...maybe I'm crying at at the #CarolBurnett50 finale with everyone singing her theme song! #ImSoGladWeHadThisTimeToget…
    {'neg': 0.177, 'neu': 0.823, 'pos': 0.0, 'compound': -0.5655}
    Mon Dec 04 03:02:00 +0000 2017
     
    RT @junerenee: A true classic and favorite!! Love #CarolBurnett50 laughed and cried...just like before!
    {'neg': 0.0, 'neu': 0.328, 'pos': 0.672, 'compound': 0.9466}
    Mon Dec 04 03:01:09 +0000 2017
     
    RT if you’ve loved reliving these classic moments during #CarolBurnett50 https://t.co/I1tIWeSlsQ
    {'neg': 0.0, 'neu': 0.719, 'pos': 0.281, 'compound': 0.5994}
    Mon Dec 04 02:51:35 +0000 2017
     
    RT @Mom101: I can't tell you the joy of seeing my 10yo laughing at this. Especially the Tim Conway dentist sketch.  #CarolBurnett50
    {'neg': 0.0, 'neu': 0.731, 'pos': 0.269, 'compound': 0.7906}
    Mon Dec 04 02:49:27 +0000 2017
     
    RT @AdeleAndMike: Thank You for bringing back such wonderful memories, the entire family would gather to watch, and we would then laugh abo…
    {'neg': 0.0, 'neu': 0.671, 'pos': 0.329, 'compound': 0.8689}
    Mon Dec 04 02:47:39 +0000 2017
     
    RT @tessab13: The Siamese Elephants skit on the Carol Burnett show is such a classic and possibly my favorite. Loved Tim Conway! #CarolBurn…
    {'neg': 0.0, 'neu': 0.736, 'pos': 0.264, 'compound': 0.8016}
    Mon Dec 04 02:43:14 +0000 2017
     
    RT @inezpanebianco: @OfficialBPeters and Carol Burnett reminiscing back on the carol burnett show is my idea of a perfect night ❤️ #CarolBu…
    {'neg': 0.0, 'neu': 0.844, 'pos': 0.156, 'compound': 0.5719}
    Mon Dec 04 02:26:30 +0000 2017
     
    What a performance by @OfficialBPeters, @KChenoweth, Carol Burnett, Steve Lawrence, and @StephenAtHome!… https://t.co/DqewQ6ASQp
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 04 02:22:42 +0000 2017
     
    RT @jonjeffryes: Bob Mackie created 65 costumes a week for The Carol Burnett Show #CarolBurnett50
    {'neg': 0.0, 'neu': 0.867, 'pos': 0.133, 'compound': 0.25}
    Mon Dec 04 02:12:59 +0000 2017
     
    RT @bmayberry92: Steve Carell with that video submission though! 😂 #carolburnett50
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 04 01:56:26 +0000 2017
     
    She’s still got it! #CarolBurnett50 https://t.co/zTxZc5zrqP
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 04 01:55:44 +0000 2017
     
    RT @BobMackie: Our friend Tim Conway never failed to make us laugh #CarolBurnett50
    {'neg': 0.0, 'neu': 0.513, 'pos': 0.487, 'compound': 0.8591}
    Mon Dec 04 01:43:24 +0000 2017
     
    RT @bpeters99: #CarolBurnett50  Am I the Only one with tears in my eyes while I give a standing ovation from my couch? One of the most bril…
    {'neg': 0.076, 'neu': 0.924, 'pos': 0.0, 'compound': -0.2263}
    Mon Dec 04 01:33:37 +0000 2017
     
    RT @SuperJen1117: This show is awesome...Some of the funniest women together on one stage 🙌🏼❤️ #CarolBurnett50
    {'neg': 0.0, 'neu': 0.806, 'pos': 0.194, 'compound': 0.5574}
    Mon Dec 04 01:24:51 +0000 2017
     
    These two are just like sisters ❤️ #CarolBurnett50 https://t.co/EPpyf2kOuV
    {'neg': 0.0, 'neu': 0.762, 'pos': 0.238, 'compound': 0.3612}
    Mon Dec 04 01:13:30 +0000 2017
     
    RT @chattie: Watching #carolburnett50  can’t quit smiling. Brings me memories of watching on Saturday nights with my mom and grandma.
    {'neg': 0.0, 'neu': 0.864, 'pos': 0.136, 'compound': 0.4588}
    Mon Dec 04 01:11:57 +0000 2017
     
    RT @DanRocks98: Settling in for the #CarolBurnett50 special. This lady is comedy gold! A true staple of my childhood.
    {'neg': 0.0, 'neu': 0.644, 'pos': 0.356, 'compound': 0.807}
    Mon Dec 04 01:05:31 +0000 2017
     
    She’s back! Tune in now to see The Carol Burnett 50th Anniversary Special on CBS and CBS All Access. #CarolBurnett50 https://t.co/HhvRmT5AR0
    {'neg': 0.0, 'neu': 0.87, 'pos': 0.13, 'compound': 0.4574}
    Mon Dec 04 01:00:09 +0000 2017
     
    Due to #NFL football overrun #CBS is delayed 11 min in: 
    Atlanta 
    Baltimore 
    Columbus
    Cleveland 
    Toledo
    Detroit
    Fli… https://t.co/oCkn11yJOR
    {'neg': 0.095, 'neu': 0.905, 'pos': 0.0, 'compound': -0.2263}
    Mon Dec 04 00:25:16 +0000 2017
     
    Relive the laughter and the magic in this star-studded tribute to everyone's favorite classic comedy show. Don't mi… https://t.co/WV6jxBnRAt
    {'neg': 0.0, 'neu': 0.648, 'pos': 0.352, 'compound': 0.8271}
    Sun Dec 03 20:00:02 +0000 2017
     
    Tonight, spend an evening together with Carol Burnett and celebrity friends as they celebrate The Carol Burnett 50t… https://t.co/uH2uM9cJFS
    {'neg': 0.0, 'neu': 0.714, 'pos': 0.286, 'compound': 0.7783}
    Sun Dec 03 18:44:14 +0000 2017
     
    Carol Burnett credits the close-knit cast for the success of her show. Join her and original cast members Vicki Law… https://t.co/1rlNPUS5EJ
    {'neg': 0.0, 'neu': 0.614, 'pos': 0.386, 'compound': 0.8658}
    Sun Dec 03 17:32:01 +0000 2017
     
    Carol Burnett and friends reflect on how the show became an instant classic that's still celebrated today! Join the… https://t.co/lZgVz9XSyM
    {'neg': 0.0, 'neu': 0.647, 'pos': 0.353, 'compound': 0.8516}
    Sun Dec 03 13:00:04 +0000 2017
     
    Tonight, celebrate with Carol Burnett, Vicki Lawrence, Lyle Waggoner &amp; more special guests! Find out how to watch T… https://t.co/Rhm6RJ34u2
    {'neg': 0.0, 'neu': 0.72, 'pos': 0.28, 'compound': 0.7897}
    Sun Dec 03 12:00:01 +0000 2017
     
    Stream Indiana at Michigan LIVE today at 12:30PM ET with a 1-month FREE trial of #CBSAllAccess. Just use promo code… https://t.co/xRem3zSyEQ
    {'neg': 0.0, 'neu': 0.825, 'pos': 0.175, 'compound': 0.6166}
    Sat Dec 02 15:55:39 +0000 2017
     
    Kristin Chenoweth (@KChenoweth) explains why audiences still crave Carol Burnett! Don't miss The Carol Burnett 50th… https://t.co/kM92NTtTfs
    {'neg': 0.0, 'neu': 0.902, 'pos': 0.098, 'compound': 0.1867}
    Sat Dec 02 14:30:03 +0000 2017
     
    Catch up on this amazing lineup of music specials! Here's how to watch Bruno Mars: 24K Magic Live at the Apollo, Th… https://t.co/Jit2CzZJCY
    {'neg': 0.0, 'neu': 0.821, 'pos': 0.179, 'compound': 0.6996}
    Sat Dec 02 13:00:07 +0000 2017
     
    Missed Bruno Mars: 24K Magic Live at the Apollo? Keep up! Watch the full show: https://t.co/itXpGB5gRm #BrunoMars https://t.co/8yyQlC1LML
    {'neg': 0.128, 'neu': 0.872, 'pos': 0.0, 'compound': -0.3595}
    Sat Dec 02 12:00:03 +0000 2017
     
    Reminisce with these classic photos and get ready to relive the magic when The Carol Burnett 50th Anniversary Speci… https://t.co/yJQEzCjoSA
    {'neg': 0.0, 'neu': 0.884, 'pos': 0.116, 'compound': 0.3612}
    Fri Dec 01 22:43:05 +0000 2017
     
    RT @LivinBiblically: In the beginning, there was Chip. And lo! Chip had a revelation. #LivingBiblically is a new comedy premiering Monday,…
    {'neg': 0.0, 'neu': 0.866, 'pos': 0.134, 'compound': 0.4199}
    Fri Dec 01 22:15:28 +0000 2017
     
    Discover these pro football stars' off-the-field talents in MVP: Most Valuable Performer, airing Jan 25 on CBS! Fin… https://t.co/g8jSRW8mQf
    {'neg': 0.0, 'neu': 0.718, 'pos': 0.282, 'compound': 0.7707}
    Fri Dec 01 21:00:12 +0000 2017
     
    Find out when #BigBrother celebrity edition, #AmazingRace, #Survivor, new drama @instinctcbs and new comedy… https://t.co/4Psl74zcCB
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 01 20:17:18 +0000 2017
     
    Actress @BethBehrs explains why Carol Burnett is her hero! Celebrate her lasting influence with The Carol Burnett 5… https://t.co/Lnt5P9jEu0
    {'neg': 0.0, 'neu': 0.691, 'pos': 0.309, 'compound': 0.8221}
    Fri Dec 01 19:32:18 +0000 2017
     
    RT @instinctcbs: New drama #Instinct, starring @Alancumming, premieres Sunday, March 11 on @CBS! https://t.co/aaL0JXpqav
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 01 18:40:34 +0000 2017
     
    RT @AmazingRaceCBS: On your marks... #AmazingRace returns Wednesday, January 3! https://t.co/XMSJSUymUZ
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 01 18:16:37 +0000 2017
     
    RT @CBSBigBrother: Mark. Your. Calendars. The celebrity edition of #BigBrother premieres February 7! https://t.co/nP3Do6jTwM
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Fri Dec 01 18:10:11 +0000 2017
     
    The Angels await you... catch up on the world's most celebrated fashion show now! Watch the full 2017… https://t.co/xfYDbEzpfX
    {'neg': 0.0, 'neu': 0.749, 'pos': 0.251, 'compound': 0.6893}
    Fri Dec 01 14:00:04 +0000 2017
     
    Comedy stars Amy Poehler, Bill Hader, @MayaRudolph &amp; @TraceeEllisRoss give the legendary Tarzan yell a try- but no… https://t.co/lrrHE2f9Nv
    {'neg': 0.0, 'neu': 0.907, 'pos': 0.093, 'compound': 0.1901}
    Fri Dec 01 12:00:03 +0000 2017
     
    Action-adventure series 'Blood &amp; Treasure' has been picked up for broadcast in Summer 2019: https://t.co/gAjqbF5bYh https://t.co/3cTBH9AKFI
    {'neg': 0.0, 'neu': 0.872, 'pos': 0.128, 'compound': 0.296}
    Fri Dec 01 01:31:30 +0000 2017
     
    Carol Burnett visited @TheTalkCBS to discuss how she got her start in show business! Watch The Carol Burnett 50th A… https://t.co/EyqmFWZHEC
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 21:10:19 +0000 2017
     
    The world's most beloved fairy tales are reimagined as a dark &amp; twisted psychological thriller in "Tell Me A Story,… https://t.co/CytIswrRj1
    {'neg': 0.0, 'neu': 0.773, 'pos': 0.227, 'compound': 0.6115}
    Thu Nov 30 20:54:39 +0000 2017
     
    RT @RansomCBS: #Ransom star @luke_j_roberts on set in Budapest for Season 2! https://t.co/rgC5ylqSMp
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 17:40:02 +0000 2017
     
    Comic @JayLeno reflects on the major impression Carol Burnett left on him. Celebrate The Carol Burnett 50th Anniver… https://t.co/PIakndHJt1
    {'neg': 0.0, 'neu': 0.752, 'pos': 0.248, 'compound': 0.6808}
    Thu Nov 30 17:30:05 +0000 2017
     
    Watch #BrunoMars perform "That's What I Like" in Bruno Mars: 24K Magic Live at the Apollo: https://t.co/hAFcB9HpE3 https://t.co/I7ZWHdDYCP
    {'neg': 0.0, 'neu': 0.865, 'pos': 0.135, 'compound': 0.3612}
    Thu Nov 30 15:00:05 +0000 2017
     
    Watch #BrunoMars perform "24K Magic" in Bruno Mars: 24K Magic Live at the Apollo: https://t.co/UgD4fC5cS6 https://t.co/iTGKJSbjwY
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 12:00:03 +0000 2017
     
    Missed Bruno Mars: 24K Magic Live at the Apollo Theater? Keep up! Watch the full show: https://t.co/itXpGB5gRm… https://t.co/nkRMIUd2Df
    {'neg': 0.128, 'neu': 0.872, 'pos': 0.0, 'compound': -0.3595}
    Thu Nov 30 07:52:09 +0000 2017
     
    RT @CWatkinsTV: That. Was. Amazing! #BrunoMars
    {'neg': 0.0, 'neu': 0.55, 'pos': 0.45, 'compound': 0.6239}
    Thu Nov 30 04:09:55 +0000 2017
     
    The stage is 🔥🔥 #BrunoMars https://t.co/nykrDH46Bs
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 04:00:27 +0000 2017
     
    RT @anitakearney65: I could watch this all night #Brunomars #BrunosTvSpecial so much fun to watch OMG
    {'neg': 0.0, 'neu': 0.796, 'pos': 0.204, 'compound': 0.5542}
    Thu Nov 30 03:59:09 +0000 2017
     
    Everybody put your hands up! #BrunoMars https://t.co/jU7NyttAqv
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 03:55:37 +0000 2017
     
    RT @Sara_nurse: My 2yr old niece heard @BrunoMars and is now up singing and dancing #BrunosTVSpecial #BrunoMars
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 03:51:38 +0000 2017
     
    RT @AllieNY86: His voice is mesmerizing 
    #BrunoMars
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 03:44:54 +0000 2017
     
    RT @BrunoMars: We love you MARY!!!! ❤️ #BrunosTvSpecial
    {'neg': 0.0, 'neu': 0.566, 'pos': 0.434, 'compound': 0.7482}
    Thu Nov 30 03:37:13 +0000 2017
     
    RT @RDMichelleLNK: Loving a dose of #BrunoMars midweek. So amazing!
    {'neg': 0.0, 'neu': 0.458, 'pos': 0.542, 'compound': 0.8513}
    Thu Nov 30 03:35:13 +0000 2017
     
    We’re only halfway into the show. Get ready for more of your favorite #BrunoMars tunes! https://t.co/BJh0PeX864
    {'neg': 0.0, 'neu': 0.698, 'pos': 0.302, 'compound': 0.7232}
    Thu Nov 30 03:33:32 +0000 2017
     
    When #BrunoMars tells you to pick up the phone, that’s exactly what you do. https://t.co/JZDS26h3lL
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 03:23:03 +0000 2017
     
    RT @MargoSlade: Best way to relax after a hard day of work watching #BrunoMars on #BrunosTVSpecial my heart is the happiest
    {'neg': 0.049, 'neu': 0.557, 'pos': 0.394, 'compound': 0.8979}
    Thu Nov 30 03:11:31 +0000 2017
     
    RT @Tiffany48184: Now that's how you open up a show!!!!! #BrunoMars
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 03:09:54 +0000 2017
     
    RT if you’re ready to party with #BrunoMars!! https://t.co/o4qQoKOzgH
    {'neg': 0.0, 'neu': 0.548, 'pos': 0.452, 'compound': 0.6988}
    Thu Nov 30 03:07:20 +0000 2017
     
    RT @BrunoMars: POP POP!!💥💥 everyone tweet #BrunosTvSpecial! We on!!!! 🍾
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 03:06:38 +0000 2017
     
    RT @Queentassia14: Bruno Mars Live at the Apollo!!!! Yeah I have to wake up at like 4 am for work.... but it's worth it. #BrunoMars #Brunos…
    {'neg': 0.0, 'neu': 0.754, 'pos': 0.246, 'compound': 0.7067}
    Thu Nov 30 03:03:15 +0000 2017
     
    RT @asbstar30: BRUNO #BrunoMars
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 03:02:34 +0000 2017
     
    Get on your feet, and prepare to dance. #BrunoMars is hitting the stage! https://t.co/yOyq8JlMZj
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Thu Nov 30 02:59:59 +0000 2017
     
    Pop pop, it's showtime! Bruno Mars: 24K Magic Live at the Apollo starts at 10/9c on CBS, or try 1 month FREE of CBS… https://t.co/cpCSYDKieM
    {'neg': 0.0, 'neu': 0.842, 'pos': 0.158, 'compound': 0.6514}
    Thu Nov 30 02:45:03 +0000 2017
     
    Calling all Bruno Mars' lovelies... don't miss Bruno Mars: 24K Magic Live at the Apollo tonight @ 10/9c on CBS &amp; CB… https://t.co/nGOgukLMx9
    {'neg': 0.0, 'neu': 0.936, 'pos': 0.064, 'compound': 0.1139}
    Thu Nov 30 00:06:05 +0000 2017
     
    You'll treasure this special night with #BrunoMars! Find out how to watch Bruno Mars: 24K Magic Live at the Apollo… https://t.co/qF2eRcJLSJ
    {'neg': 0.0, 'neu': 0.772, 'pos': 0.228, 'compound': 0.6825}
    Wed Nov 29 23:20:13 +0000 2017
     
    RT @BrunoMars: Ladies put your hoops on tonight! Fellas break out that good cologne! #BrunosTvSpecial is airing on @CBS! What y’all trynna…
    {'neg': 0.0, 'neu': 0.848, 'pos': 0.152, 'compound': 0.5826}
    Wed Nov 29 23:15:24 +0000 2017
     
    RT @YoungSheldon: Sheldon will do whatever it takes to get his hands on his own computer in tomorrow's all-new episode of #YoungSheldon! ht…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Wed Nov 29 23:14:57 +0000 2017
     
    RT @survivorcbs: What drama will ensue on back-to-back episodes of #Survivor tonight? Watch a sneak peek now: https://t.co/LHcxyClSw1 https…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Wed Nov 29 23:13:40 +0000 2017
     
    Discover every reason why #BrunoMars is 24K Magic! Watch Bruno Mars: 24K Magic Live at the Apollo tonight @ 10/9c o… https://t.co/oaPjvl3rSW
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Wed Nov 29 22:34:56 +0000 2017
     
    Did you catch these tiny moments during the 2017 Victoria's Secret Fashion Show? Find out what you may have missed:… https://t.co/IIyupot9uU
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Wed Nov 29 21:07:33 +0000 2017
     
    RT @funnyordie: Live! Watch Tim Meadows and Patrick Brammall of @NoActivityCBS play themselves in new 8-bit video game! https://t.co/nXCuHO…
    {'neg': 0.0, 'neu': 0.858, 'pos': 0.142, 'compound': 0.4559}
    Wed Nov 29 20:41:13 +0000 2017
     
    The 'Only Angel' @Harry_Styles saw last night were the ones on the runway! Watch him perform at the #VSFashionShow:… https://t.co/na8GAiZwmS
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Wed Nov 29 15:00:07 +0000 2017
     
    This entire collection is drop-dead gorgeous. See all the most sizzling looks from the 2017 #VSFashionShow:… https://t.co/zpTKXcm05r
    {'neg': 0.0, 'neu': 0.8, 'pos': 0.2, 'compound': 0.6124}
    Wed Nov 29 12:00:06 +0000 2017
     
    Stars @Miguel, @Harry_Styles, @leslieodomjr &amp; more set the tone for a truly incredible show! Watch all the musical… https://t.co/kmo4l3aGai
    {'neg': 0.0, 'neu': 0.842, 'pos': 0.158, 'compound': 0.4926}
    Wed Nov 29 09:09:48 +0000 2017
     
    Some of the best moments happen behind the scenes! Here's your backstage pass to the 2017 Victoria's Secret Fashion… https://t.co/S8MpOnhC6b
    {'neg': 0.0, 'neu': 0.809, 'pos': 0.191, 'compound': 0.6696}
    Wed Nov 29 08:39:05 +0000 2017
     
    From the sky-high heels to the jaw-dropping wings, these outfits are fashion goals! See all the best looks from the… https://t.co/GnyZ8NnBYH
    {'neg': 0.0, 'neu': 0.817, 'pos': 0.183, 'compound': 0.6696}
    Wed Nov 29 08:22:35 +0000 2017
     
    Missed the world’s most celebrated fashion show? Catch up on your own time and watch the entire 2017 Victoria's Sec… https://t.co/9OuYvHItzh
    {'neg': 0.087, 'neu': 0.754, 'pos': 0.158, 'compound': 0.4201}
    Wed Nov 29 07:56:05 +0000 2017
     
    RT @VictoriasSecret: That’s a wrap on the 2017 #VSFashionShow!! BIG thanks to the Angels, the people of Shanghai &amp; YOU for watching! https:…
    {'neg': 0.0, 'neu': 0.848, 'pos': 0.152, 'compound': 0.5826}
    Wed Nov 29 06:41:12 +0000 2017
     
    Want to relive all of your favorite #VSFashionShow runway moments?  @GIPHY has you covered. https://t.co/mgfziAQR4c
    {'neg': 0.0, 'neu': 0.751, 'pos': 0.249, 'compound': 0.5106}
    Wed Nov 29 04:13:05 +0000 2017
     
    RT @vannalovescats: Another successful show. I’m so proud of all of the beautiful ladies, they all did amazing. The musical guests were FAN…
    {'neg': 0.0, 'neu': 0.551, 'pos': 0.449, 'compound': 0.9476}
    Wed Nov 29 04:02:16 +0000 2017
     
    55 Models. 88 Looks. 1 FABULOUS Night. #VSFashionShow https://t.co/TmYqRjDRE5
    {'neg': 0.0, 'neu': 0.629, 'pos': 0.371, 'compound': 0.6289}
    Wed Nov 29 04:00:58 +0000 2017
     
    Two. Million. Dollar. Bra. 😮  #VSFashionShow https://t.co/4dp7RbMHlp
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Wed Nov 29 03:57:07 +0000 2017
     



```python
cbs_complete_df = pd.DataFrame({'Date (CBS)': cbs_date_list, 'Tweet (CBS)': cbs_text_list, 
                                'Negative Score (CBS)': cbs_negative_list, 'Neutral Score (CBS)': cbs_neutral_list, 
                                'Positive Score (CBS)': cbs_positive_list, 'Compound Score (CBS)': cbs_compound})
cbs_final_df = cbs_complete_df[['Date (CBS)', 'Tweet (CBS)', 'Negative Score (CBS)', 'Neutral Score (CBS)', 
                 'Positive Score (CBS)', 'Compound Score (CBS)']]
cbs_final_df.head()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date (CBS)</th>
      <th>Tweet (CBS)</th>
      <th>Negative Score (CBS)</th>
      <th>Neutral Score (CBS)</th>
      <th>Positive Score (CBS)</th>
      <th>Compound Score (CBS)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mon Dec 11 00:26:50 +0000 2017</td>
      <td>Due to NFL overrun, CBS is delayed 8 mins in t...</td>
      <td>0.083</td>
      <td>0.917</td>
      <td>0.000</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mon Dec 11 00:25:54 +0000 2017</td>
      <td>Due to NFL overrun CBS is delayed 7 mins in th...</td>
      <td>0.087</td>
      <td>0.913</td>
      <td>0.000</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sun Dec 10 22:49:02 +0000 2017</td>
      <td>RT @NoActivityCBS: If you want the intel, you ...</td>
      <td>0.000</td>
      <td>0.936</td>
      <td>0.064</td>
      <td>0.0772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sun Dec 10 22:48:46 +0000 2017</td>
      <td>RT @startrekcbs: .@albinokid and @wcruz73 are ...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sat Dec 09 18:24:37 +0000 2017</td>
      <td>Don’t miss America’s Game! Stream the Army-Nav...</td>
      <td>0.070</td>
      <td>0.742</td>
      <td>0.189</td>
      <td>0.5754</td>
    </tr>
  </tbody>
</table>
</div>




```python
cbs_compound_df = pd.DataFrame(cbs_compound)
cbs_compound_df.columns = ['CBS Compound Score']
cbs_compound_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CBS Compound Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.5754</td>
    </tr>
  </tbody>
</table>
</div>




```python
#CNN

cnn_text_list = []
cnn_date_list = []
cnn_negative_list = []
cnn_neutral_list = []
cnn_positive_list = []
cnn_compound = []


cnn_public_tweets = api.user_timeline(cnn, count = 100)

for cnn_tweet in cnn_public_tweets:
    
    cnn_text = cnn_tweet['text']
    cnn_date = cnn_tweet['created_at']
    
    print(cnn_text)
    
    
    cnn_scores = analyzer.polarity_scores(cnn_text)
    print(cnn_scores)
    print(cnn_date)
    print(' ')
    
    cnn_text_list.append(cnn_text)
    cnn_date_list.append(cnn_date)
    cnn_negative_list.append(cnn_scores['neg'])
    cnn_neutral_list.append(cnn_scores['neu'])
    cnn_positive_list.append(cnn_scores['pos'])
    cnn_compound.append(cnn_scores['compound'])
    
```

    Democratic lawmakers have asked the Treasury Department for documents on financial dealings with Russia… https://t.co/pBmTUHxsAz
    {'neg': 0.0, 'neu': 0.886, 'pos': 0.114, 'compound': 0.2023}
    Mon Dec 11 04:00:21 +0000 2017
     
    McLaren's 'most extreme' road car costs $1 million https://t.co/a1Unk82nwx https://t.co/yvlL9LCdSq
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 03:50:08 +0000 2017
     
    Venezuelan President Nicolas Maduro has said that some of the main opposition parties will not be allowed to run in… https://t.co/VmN9pU5gEC
    {'neg': 0.0, 'neu': 0.881, 'pos': 0.119, 'compound': 0.4019}
    Mon Dec 11 03:40:30 +0000 2017
     
    These are the most eye-catching photos from 2017 https://t.co/lXgXsyNfa6 https://t.co/Z6N0yRLk60
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 03:30:12 +0000 2017
     
    Heavy snow fell in many parts of the UK on Sunday as Storm Caroline, the biggest storm so far this year, caused wid… https://t.co/R9UCEICrxm
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 03:25:06 +0000 2017
     
    Global arms sales increased for the first time in five years in 2016, a new report says https://t.co/thhQ2sqH3c https://t.co/mjSUBUVzic
    {'neg': 0.0, 'neu': 0.89, 'pos': 0.11, 'compound': 0.2732}
    Mon Dec 11 03:17:04 +0000 2017
     
    The revelation that a black hole grew in just five hundred million years challenges our models of how early galaxie… https://t.co/A6dhPi6bDe
    {'neg': 0.0, 'neu': 0.936, 'pos': 0.064, 'compound': 0.0772}
    Mon Dec 11 03:05:48 +0000 2017
     
    Voting for CNN Hero of the Year CLOSES Tuesday 12/12 at 11:59pm PT -- VOTE NOW at https://t.co/ANMKKkTX4Z!… https://t.co/dx1AFkowA7
    {'neg': 0.0, 'neu': 0.822, 'pos': 0.178, 'compound': 0.5983}
    Mon Dec 11 03:01:13 +0000 2017
     
    Sunday was a tale of two campaigns for Alabama’s special Senate election, with Democratic candidate Doug Jones barn… https://t.co/jL6LWnNYrp
    {'neg': 0.0, 'neu': 0.863, 'pos': 0.137, 'compound': 0.4019}
    Mon Dec 11 02:56:52 +0000 2017
     
    "Time is of the essence," UN official says after visit to North Korea https://t.co/Ndbay5jZKB https://t.co/Ik94KKGF5X
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 02:47:04 +0000 2017
     
    "Mutual destruction is only one impulsive tantrum away." 2017 Nobel Peace Prize winners urge world leaders to prohi… https://t.co/cdt0zVvUGy
    {'neg': 0.214, 'neu': 0.461, 'pos': 0.326, 'compound': 0.5267}
    Mon Dec 11 02:36:11 +0000 2017
     
    3-D printing braces up sea turtle with gap in its shell https://t.co/mhGanIyPvv https://t.co/sfqprWjpiv
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 02:30:44 +0000 2017
     
    Turkish President Recep Tayyip Erdogan called Israel a "terrorist" and "child-murderer state" as he criticized US P… https://t.co/BKnHlFnpqF
    {'neg': 0.135, 'neu': 0.865, 'pos': 0.0, 'compound': -0.3612}
    Mon Dec 11 02:17:37 +0000 2017
     
    A Hong Kong resident shares his tips for street photography so your images will look like a pro snapped them… https://t.co/5WoufYP4mX
    {'neg': 0.0, 'neu': 0.783, 'pos': 0.217, 'compound': 0.5719}
    Mon Dec 11 02:14:06 +0000 2017
     
    He was shot and killed by Chicago police in 2014. But that’s just the beginning of this story. #BeneathTheSkin… https://t.co/5v9LwaIdhK
    {'neg': 0.191, 'neu': 0.809, 'pos': 0.0, 'compound': -0.6705}
    Mon Dec 11 02:00:23 +0000 2017
     
    The Chicago Board Options Exchange on Sunday began trading bitcoin futures -- a new step for the cryptocurrency… https://t.co/ioVnnfy3wf
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 01:51:58 +0000 2017
     
    17 million infants around the world are breathing toxic air that can severely affect brain development, a new UNICE… https://t.co/17hus7Dff3
    {'neg': 0.143, 'neu': 0.857, 'pos': 0.0, 'compound': -0.4588}
    Mon Dec 11 01:30:14 +0000 2017
     
    Global arms sales increased for the first time in five years in 2016, a new report says https://t.co/K6yTDlfed9 https://t.co/i4cYJ3hftu
    {'neg': 0.0, 'neu': 0.89, 'pos': 0.11, 'compound': 0.2732}
    Mon Dec 11 01:21:13 +0000 2017
     
    Israeli army destroys tunnel from Gaza https://t.co/emKGf8IPZW https://t.co/OHXDWt8Vtc
    {'neg': 0.34, 'neu': 0.66, 'pos': 0.0, 'compound': -0.5574}
    Mon Dec 11 00:45:05 +0000 2017
     
    During an appearance on a conspiracy-driven radio show in 2011, Alabama Senate candidate Roy Moore said getting rid… https://t.co/B4cZRMxXO6
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 00:30:08 +0000 2017
     
    Sirius XM faces celebrity backlash after former White House Chief Strategist Steve Bannon rejoins radio show… https://t.co/HLJZVwW78o
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 00:15:05 +0000 2017
     
    The rise of a ‘fish mafia’ in this small Mexican town is killing off an entire population of small porpoises.… https://t.co/W0tTmNUogp
    {'neg': 0.188, 'neu': 0.812, 'pos': 0.0, 'compound': -0.6597}
    Mon Dec 11 00:00:41 +0000 2017
     
    “We're not looking for perfection, we're just looking for better than yesterday." After rehab changed his life, Aar… https://t.co/fVpZmBWzqm
    {'neg': 0.131, 'neu': 0.742, 'pos': 0.127, 'compound': -0.0253}
    Sun Dec 10 23:46:37 +0000 2017
     
    President Trump spoke at the opening of the Mississippi Civil Rights Museum. Some civil rights and community leader… https://t.co/sX8XMVX9k3
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 23:30:43 +0000 2017
     
    Israel airstrikes, Gaza rockets amid tensions over Jerusalem https://t.co/8zRuyw7PO4 https://t.co/P47G7huOe0
    {'neg': 0.231, 'neu': 0.769, 'pos': 0.0, 'compound': -0.4019}
    Sun Dec 10 23:15:08 +0000 2017
     
    FBI email shows Attorney General Jeff Sessions wasn't required to disclose foreign contacts for security clearance… https://t.co/RWekrGKgQJ
    {'neg': 0.0, 'neu': 0.87, 'pos': 0.13, 'compound': 0.34}
    Sun Dec 10 23:00:07 +0000 2017
     
    McLaren's 'most extreme' road car costs $1 million https://t.co/5CwE92eNfD https://t.co/GrQmRbIrMg
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:45:07 +0000 2017
     
    Vice President Mike Pence and Palestinian Authority President Mahmoud Abbas won't meet, vice president's office con… https://t.co/Tx9YkRLoca
    {'neg': 0.0, 'neu': 0.925, 'pos': 0.075, 'compound': 0.0772}
    Sun Dec 10 22:30:10 +0000 2017
     
    UN chief says "America First" slogan is "detrimental to American interests" https://t.co/Fdmv3o9TUT https://t.co/T3Z51xTtC2
    {'neg': 0.0, 'neu': 0.857, 'pos': 0.143, 'compound': 0.25}
    Sun Dec 10 22:15:09 +0000 2017
     
    Even with Congress in session, all eyes will be on Alabama on Tuesday https://t.co/Qa1pMjzwfv https://t.co/PjNhNKkHcH
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:00:20 +0000 2017
     
    Interior Secretary Ryan Zinke is pushing for a controversial road through a federally protected wilderness area in… https://t.co/pDolVX4lDn
    {'neg': 0.096, 'neu': 0.749, 'pos': 0.155, 'compound': 0.2732}
    Sun Dec 10 21:45:11 +0000 2017
     
    "We should all be willing to listen to" women who speak up about inappropriate sexual behavior, including President… https://t.co/BtPynSIb5U
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 21:30:06 +0000 2017
     
    The winners of the 2017 Nobel Peace Prize warn that nuclear destruction is 'one impulsive tantrum away'… https://t.co/xzz1xLGCSw
    {'neg': 0.265, 'neu': 0.403, 'pos': 0.332, 'compound': 0.4588}
    Sun Dec 10 21:15:06 +0000 2017
     
    "It's not at all a surprise that they (Russians) would make outreach to people they thought would have influence in… https://t.co/24yiPExN9C
    {'neg': 0.083, 'neu': 0.821, 'pos': 0.096, 'compound': 0.0736}
    Sun Dec 10 21:00:42 +0000 2017
     
    Republican Sen. Susan Collins says she's not sure if she'll support the tax reform plan https://t.co/5YSNF3fLne https://t.co/tnM55Uu9G2
    {'neg': 0.1, 'neu': 0.763, 'pos': 0.137, 'compound': 0.1872}
    Sun Dec 10 20:45:14 +0000 2017
     
    Take a look at this week in politics https://t.co/8ktzXSEPN4 https://t.co/i8OYgFjIPJ
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:30:09 +0000 2017
     
    Futures trading is a new step for Bitcoin's wild ride https://t.co/GMHpH5Fw8z https://t.co/TmNf8tzIf8
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:15:03 +0000 2017
     
    She lost her leg after being hit by a drunk driver. Now she empowers fellow amputees to overcome their physical lim… https://t.co/dINDZ5JW69
    {'neg': 0.198, 'neu': 0.802, 'pos': 0.0, 'compound': -0.5719}
    Sun Dec 10 20:01:56 +0000 2017
     
    Mix-and-match health coverage can be a risky alternative to Obamacare https://t.co/23NHVKbXQW https://t.co/76jyYlu0lj
    {'neg': 0.153, 'neu': 0.847, 'pos': 0.0, 'compound': -0.2023}
    Sun Dec 10 19:45:05 +0000 2017
     
    CORRECTION: Jailed "King of Spin" Max Clifford dies in UK hospital https://t.co/mDlPZdNE9K https://t.co/qgHMGBdTP3
    {'neg': 0.211, 'neu': 0.789, 'pos': 0.0, 'compound': -0.4939}
    Sun Dec 10 19:41:08 +0000 2017
     
    Swedish police question suspects after synagogue attack https://t.co/3AlN0ZnOnc https://t.co/5Enu6RVoqw
    {'neg': 0.44, 'neu': 0.56, 'pos': 0.0, 'compound': -0.6705}
    Sun Dec 10 19:30:29 +0000 2017
     
    President Trump has to live with the consequences of his Israel decision, writes Nic Robertson for @CNNOpinion… https://t.co/QbOk7FJSwi
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 19:15:12 +0000 2017
     
    Alabama Senate candidate Roy Moore says he has never molested anyone https://t.co/4F7ipmrsLz https://t.co/quVFW9P8gs
    {'neg': 0.0, 'neu': 0.833, 'pos': 0.167, 'compound': 0.3412}
    Sun Dec 10 19:01:09 +0000 2017
     
    "Time is of the essence," UN official says after visit to North Korea https://t.co/kPzalGuqWh https://t.co/KHWyssfkeX
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 18:45:08 +0000 2017
     
    The hidden fear driving the market surge https://t.co/jZVgAGZ7it https://t.co/MADYFBCHZ3
    {'neg': 0.286, 'neu': 0.714, 'pos': 0.0, 'compound': -0.4939}
    Sun Dec 10 18:30:10 +0000 2017
     
    Protesters and police clash outside US embassy in Beirut https://t.co/buz5mXWp3F https://t.co/X2JNZ8p3wP
    {'neg': 0.16, 'neu': 0.84, 'pos': 0.0, 'compound': -0.2263}
    Sun Dec 10 18:15:07 +0000 2017
     
    Heavy snow brings travel chaos to UK https://t.co/vVRzn8LEVP https://t.co/Mm7FLshDpW
    {'neg': 0.316, 'neu': 0.684, 'pos': 0.0, 'compound': -0.5719}
    Sun Dec 10 18:00:09 +0000 2017
     
    Ski racer Lindsey Vonn injures back and pulls out of a competition in St. Moritz https://t.co/5zVmooeTLI https://t.co/KjveQNCQFs
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 17:45:10 +0000 2017
     
    UK foreign minister ends Iran trip with no announcement on jailed woman https://t.co/9wBvAfh8Qr https://t.co/2Wgp4dmdSE
    {'neg': 0.31, 'neu': 0.69, 'pos': 0.0, 'compound': -0.6597}
    Sun Dec 10 17:30:07 +0000 2017
     
    The gold rush for black market fish bladders has pushed this tiny porpoise toward extinction #vaquita… https://t.co/TQfbYwqPZc
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 17:15:33 +0000 2017
     
    Voting for CNN Hero of the Year CLOSES Tuesday 12/12 at 11:59pm PT -- VOTE NOW at https://t.co/ANMKKlbytz!… https://t.co/CuFQv7zB4U
    {'neg': 0.0, 'neu': 0.822, 'pos': 0.178, 'compound': 0.5983}
    Sun Dec 10 17:03:05 +0000 2017
     
    Here's your ski resort guide to St. Moritz, Switzerland https://t.co/9NcNA31QQ1 https://t.co/57kMyQRIOa
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 16:51:30 +0000 2017
     
    "King of Spin" Max Clifford dies in UK jail https://t.co/mDlPZdNE9K https://t.co/XodLjZtMLQ
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 16:30:09 +0000 2017
     
    Alabama Sen. Richard Shelby says he "couldn't vote for Roy Moore" https://t.co/3T8Xq0vCXe https://t.co/mIZijkBue1
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 16:20:39 +0000 2017
     
    What a shot! The most amazing sports photos of 2017 https://t.co/bhSvDBYijR https://t.co/P8fBoXNPf2
    {'neg': 0.0, 'neu': 0.695, 'pos': 0.305, 'compound': 0.6581}
    Sun Dec 10 16:00:24 +0000 2017
     
    UN Chief António Guterres: "It's essential to understand that refugees are not terrorists. They are the first victi… https://t.co/x5MU0FygNd
    {'neg': 0.0, 'neu': 0.845, 'pos': 0.155, 'compound': 0.5096}
    Sun Dec 10 15:57:01 +0000 2017
     
    Pressure is mounting on Congress to act before year’s end on Deferred Action for Childhood Arrivals. What they deci… https://t.co/AwRUbU4BJb
    {'neg': 0.104, 'neu': 0.896, 'pos': 0.0, 'compound': -0.296}
    Sun Dec 10 15:45:09 +0000 2017
     
    "It is very important for the world...that the US engages," UN Chief Antonio Guterres tells @FareedZakaria during a… https://t.co/FDF9ehtiri
    {'neg': 0.0, 'neu': 0.896, 'pos': 0.104, 'compound': 0.2716}
    Sun Dec 10 15:43:09 +0000 2017
     
    The Honest Company founder Jessica Alba tells @PoppyHarlowCNN that "it’s tough when you’re the only woman in the ro… https://t.co/14mpfoLC3A
    {'neg': 0.066, 'neu': 0.789, 'pos': 0.145, 'compound': 0.4215}
    Sun Dec 10 15:30:19 +0000 2017
     
    "This is kind of the new normal," says California Gov. Jerry Brown, after a week of raging wildfires… https://t.co/Sk6Yh2IffQ
    {'neg': 0.167, 'neu': 0.833, 'pos': 0.0, 'compound': -0.5267}
    Sun Dec 10 15:16:21 +0000 2017
     
    She lost her leg after being hit by a drunk driver. Now she empowers fellow amputees to overcome their physical lim… https://t.co/EVxCuS4uQY
    {'neg': 0.198, 'neu': 0.802, 'pos': 0.0, 'compound': -0.5719}
    Sun Dec 10 15:02:54 +0000 2017
     
    A group of precocious kids had some tough questions for Santa Claus on @nbcsnl last night https://t.co/M70oxgiqRJ https://t.co/zDbbuNDoUT
    {'neg': 0.086, 'neu': 0.914, 'pos': 0.0, 'compound': -0.128}
    Sun Dec 10 14:50:49 +0000 2017
     
    Rep. Adam Schiff on Russian outreach to Hope Hicks: “It’s not at all a surprise that they would make outreach to pe… https://t.co/MwQM9RE14r
    {'neg': 0.067, 'neu': 0.669, 'pos': 0.264, 'compound': 0.647}
    Sun Dec 10 14:38:22 +0000 2017
     
    Instagram is testing a standalone messaging app called Direct https://t.co/0B9ky2jDio https://t.co/EIP9KH2gmO
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 14:30:19 +0000 2017
     
    US Ambassador to the United Nations Nikki Haley: "Jerusalem is the capital of Israel...whatever is East Jerusalem o… https://t.co/hnnw3xaf3q
    {'neg': 0.0, 'neu': 0.865, 'pos': 0.135, 'compound': 0.4215}
    Sun Dec 10 14:28:37 +0000 2017
     
    US Ambassador to the United Nations Nikki Haley: “I have no concern” that President Trump’s Jerusalem decision will… https://t.co/ZHep68YyhL
    {'neg': 0.1, 'neu': 0.773, 'pos': 0.127, 'compound': 0.1531}
    Sun Dec 10 14:28:16 +0000 2017
     
    US Ambassador to the United Nations Nikki Haley says moving the capital of Israel to Jerusalem will "move the ball… https://t.co/DbDpLD6McJ
    {'neg': 0.0, 'neu': 0.877, 'pos': 0.123, 'compound': 0.4215}
    Sun Dec 10 14:22:03 +0000 2017
     
    Republican Alabama Sen. Richard Shelby says there's "something for just about everybody" in the GOP tax bill… https://t.co/hy29AebKKk
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 14:13:18 +0000 2017
     
    “I want to reiterate again: I didn’t vote for Roy Moore, I wouldn’t vote for Roy Moore, I think the Republican Part… https://t.co/2R9DV9eqAo
    {'neg': 0.0, 'neu': 0.936, 'pos': 0.064, 'compound': 0.0772}
    Sun Dec 10 14:12:33 +0000 2017
     
    Republican Alabama Sen. Richard Shelby on Roy Moore’s accusers: “I think the women are believable, I have no reason… https://t.co/cVbHjeOmTy
    {'neg': 0.109, 'neu': 0.891, 'pos': 0.0, 'compound': -0.296}
    Sun Dec 10 14:08:02 +0000 2017
     
    Republican Alabama Sen. Richard Shelby: “I couldn’t vote for Roy Moore, I didn’t vote for Roy Moore, I wrote in a d… https://t.co/Q40njVbgsQ
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 14:06:18 +0000 2017
     
    An Iranian-Turkish gold trader admitted in a US court that he made "maybe $150 million" by helping Iran dodge inter… https://t.co/yg1Kck4Zxt
    {'neg': 0.0, 'neu': 0.833, 'pos': 0.167, 'compound': 0.3818}
    Sun Dec 10 14:01:06 +0000 2017
     
    The rise of a ‘fish mafia’ in this small Mexican town is killing off an entire population of small porpoises.… https://t.co/AmToMyq64y
    {'neg': 0.188, 'neu': 0.812, 'pos': 0.0, 'compound': -0.6597}
    Sun Dec 10 13:30:20 +0000 2017
     
    We all tend to change our voices when talking to babies, but those changes can also vary across cultures. Here's ho… https://t.co/knCYMFE4kH
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 13:00:24 +0000 2017
     
    “We're not looking for perfection, we're just looking for better than yesterday." After rehab changed his life, Aar… https://t.co/5HRacrBX3M
    {'neg': 0.131, 'neu': 0.742, 'pos': 0.127, 'compound': -0.0253}
    Sun Dec 10 12:31:30 +0000 2017
     
    Four questions the November jobs report will help to answer:
    
    - How low can unemployment go?
    
    - Any remaining hurri… https://t.co/3mVntmTOA4
    {'neg': 0.211, 'neu': 0.675, 'pos': 0.114, 'compound': -0.3182}
    Sun Dec 10 11:30:04 +0000 2017
     
    Dramatic footage from @sarasidnerCNN shows one of the Southern California wildfires has jumped over a freeway while… https://t.co/DFuXTPkuja
    {'neg': 0.0, 'neu': 0.93, 'pos': 0.07, 'compound': 0.0516}
    Sun Dec 10 11:01:40 +0000 2017
     
    New imaging finds crescent-shaped eye damage in a woman who tried to view the total solar eclipse in August… https://t.co/SdkIu6dPPF
    {'neg': 0.151, 'neu': 0.849, 'pos': 0.0, 'compound': -0.4939}
    Sun Dec 10 10:30:12 +0000 2017
     
    The stock market has seen a record-breaking ascent this year. But who is actually making money off the boom? Rich p… https://t.co/CJ8V8KtDJb
    {'neg': 0.0, 'neu': 0.847, 'pos': 0.153, 'compound': 0.5574}
    Sun Dec 10 10:00:09 +0000 2017
     
    Flu season is gaining speed in the United States, especially in the South, after getting off to a slow start… https://t.co/nFotP1jG5X
    {'neg': 0.103, 'neu': 0.675, 'pos': 0.222, 'compound': 0.4588}
    Sun Dec 10 09:30:17 +0000 2017
     
    The gold rush for black market fish bladders has pushed this tiny porpoise toward extinction #vaquita… https://t.co/5Wn1UGNZuC
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 09:00:37 +0000 2017
     
    US jobs continued their strong run in November as employers added 228,000 jobs and the unemployment rate remained a… https://t.co/NYSqgRvB0s
    {'neg': 0.12, 'neu': 0.744, 'pos': 0.136, 'compound': 0.1027}
    Sun Dec 10 08:30:15 +0000 2017
     
    "Horse people are a community... Leaving one behind is like leaving a child behind ... you just can't do it."
    
    Hors… https://t.co/6Y888FVc4X
    {'neg': 0.0, 'neu': 0.884, 'pos': 0.116, 'compound': 0.3612}
    Sun Dec 10 08:00:19 +0000 2017
     
    How states that carried Trump to victory avoid one sticky part of the tax bill https://t.co/WJ7mEeH7TA https://t.co/NYXJI8onFY
    {'neg': 0.121, 'neu': 0.879, 'pos': 0.0, 'compound': -0.296}
    Sun Dec 10 07:30:05 +0000 2017
     
    State Department officials defended President Trump's recognition of Jerusalem as Israel's capital, saying it refle… https://t.co/MfvgXGJ45B
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 07:00:16 +0000 2017
     
    A woman who made it through a wildfire in Southern California turned her home into a donation site for victims… https://t.co/psGRaWVtke
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 06:30:08 +0000 2017
     
    Hispanic unemployment in the US is at an all-time low under President Trump https://t.co/pp37gMuqQR https://t.co/W0DQM5JLSY
    {'neg': 0.278, 'neu': 0.722, 'pos': 0.0, 'compound': -0.6124}
    Sun Dec 10 06:00:28 +0000 2017
     
    "My life depends on it. I need you to make your vote match your principles, Senator." A man who says he was recentl… https://t.co/i8HgQFVMXl
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 05:31:46 +0000 2017
     
    Facebook has made its internal sexual harassment policy public https://t.co/Mnhn3IkTo9 https://t.co/mb3KpYAhZa
    {'neg': 0.259, 'neu': 0.741, 'pos': 0.0, 'compound': -0.5423}
    Sun Dec 10 05:01:09 +0000 2017
     
    A @CNNMoney reporter bought $250 in bitcoin. Here's what he learned https://t.co/AXbsPWYfWQ https://t.co/OF973bs3zG
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 04:30:04 +0000 2017
     
    References to climate change and the EPA's use of renewable energy have been removed from several of its web pages,… https://t.co/f4IiBEOLx2
    {'neg': 0.0, 'neu': 0.905, 'pos': 0.095, 'compound': 0.2732}
    Sun Dec 10 04:01:07 +0000 2017
     
    The critically acclaimed drama "Big Little Lies" is officially coming back for another season… https://t.co/WmE6oKMtVQ
    {'neg': 0.152, 'neu': 0.848, 'pos': 0.0, 'compound': -0.3626}
    Sun Dec 10 03:45:04 +0000 2017
     
    The rise of a ‘fish mafia’ in this small Mexican town is killing off an entire population of small porpoises.… https://t.co/RE2Aj4GJtB
    {'neg': 0.188, 'neu': 0.812, 'pos': 0.0, 'compound': -0.6597}
    Sun Dec 10 03:30:31 +0000 2017
     
    The official death toll in Puerto Rico from Hurricane Maria rises to 64, public safety agency says. Investigations… https://t.co/IM9AcX6okY
    {'neg': 0.165, 'neu': 0.717, 'pos': 0.118, 'compound': -0.2732}
    Sun Dec 10 03:16:05 +0000 2017
     
    The European Union has blacklisted 17 countries and territories as part of a crackdown on tax havens… https://t.co/hJGEYT8o0v
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 03:15:08 +0000 2017
     
    “It’s more than a cup of coffee, it’s a human rights movement” says Amy Wright, who founded @bittyandbeauscoffee em… https://t.co/PswOGnrE1K
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 03:03:13 +0000 2017
     
    Obama is right — US democracy is fragile, writes Ruth Ben-Ghiat https://t.co/3HT3bbHVu6 (via @CNNOpinion) https://t.co/QNX9n7aDvC
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 02:45:06 +0000 2017
     
    For the first time in American politics, anonymous "dark money" political donations could become tax-deductible… https://t.co/8vx7gDtlKu
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 02:30:17 +0000 2017
     
    University of Oklahoma quarterback Baker Mayfield wins the Heisman Trophy, as college football's best player… https://t.co/e9BpQH8fDi
    {'neg': 0.0, 'neu': 0.639, 'pos': 0.361, 'compound': 0.836}
    Sun Dec 10 02:21:56 +0000 2017
     
    Bitcoin's phenomenal rise this year may be making speculators very rich, but some observers say it's terrible for t… https://t.co/RJ1TEaPxQR
    {'neg': 0.169, 'neu': 0.732, 'pos': 0.099, 'compound': -0.4026}
    Sun Dec 10 02:15:02 +0000 2017
     



```python
cnn_complete_df = pd.DataFrame({'Date (CNN)': cnn_date_list, 'Tweet (CNN)': cnn_text_list, 
                                'Negative Score (CNN)': cnn_negative_list, 'Neutral Score (CNN)': cnn_neutral_list, 
                                'Positive Score (CNN)': cnn_positive_list, 'Compound Score (CNN)': cnn_compound})
cnn_final_df = cnn_complete_df[['Date (CNN)', 'Tweet (CNN)', 'Negative Score (CNN)', 'Neutral Score (CNN)', 
                 'Positive Score (CNN)', 'Compound Score (CNN)']]
cnn_final_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date (CNN)</th>
      <th>Tweet (CNN)</th>
      <th>Negative Score (CNN)</th>
      <th>Neutral Score (CNN)</th>
      <th>Positive Score (CNN)</th>
      <th>Compound Score (CNN)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mon Dec 11 04:00:21 +0000 2017</td>
      <td>Democratic lawmakers have asked the Treasury D...</td>
      <td>0.0</td>
      <td>0.886</td>
      <td>0.114</td>
      <td>0.2023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mon Dec 11 03:50:08 +0000 2017</td>
      <td>McLaren's 'most extreme' road car costs $1 mil...</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon Dec 11 03:40:30 +0000 2017</td>
      <td>Venezuelan President Nicolas Maduro has said t...</td>
      <td>0.0</td>
      <td>0.881</td>
      <td>0.119</td>
      <td>0.4019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mon Dec 11 03:30:12 +0000 2017</td>
      <td>These are the most eye-catching photos from 20...</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mon Dec 11 03:25:06 +0000 2017</td>
      <td>Heavy snow fell in many parts of the UK on Sun...</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cnn_compound_df = pd.DataFrame(cnn_compound)
cnn_compound_df.columns = ['CNN Compound Score']
cnn_compound_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNN Compound Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.2023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.4019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#FOX

fox_text_list = []
fox_date_list = []
fox_negative_list = []
fox_neutral_list = []
fox_positive_list = []
fox_compound = []


fox_public_tweets = api.user_timeline(fox, count = 100)

for fox_tweet in fox_public_tweets:
    
    fox_text = fox_tweet['text']
    fox_date = fox_tweet['created_at']
    
    print(fox_text)
    
    
    fox_scores = analyzer.polarity_scores(fox_text)
    print(fox_scores)
    print(fox_date)
    print(' ')
    
    fox_text_list.append(fox_text)
    fox_date_list.append(fox_date)
    fox_negative_list.append(fox_scores['neg'])
    fox_neutral_list.append(fox_scores['neu'])
    fox_positive_list.append(fox_scores['pos'])
    fox_compound.append(fox_scores['compound'])
    
```

    On @ffweekend, @SheriffClarke slammed Rep. John Lewis for boycotting @POTUS's appearance at a Civil Rights museum i… https://t.co/29ZcytSMHN
    {'neg': 0.144, 'neu': 0.856, 'pos': 0.0, 'compound': -0.4019}
    Mon Dec 11 04:17:05 +0000 2017
     
    In his speech at the opening of the Mississippi Civil Rights Museum, @POTUS called for "a future of freedom, equali… https://t.co/CZfZZ60T4a
    {'neg': 0.0, 'neu': 0.826, 'pos': 0.174, 'compound': 0.6369}
    Mon Dec 11 04:15:02 +0000 2017
     
    .@JesseBWatters: "The @FBI, the Department of Justice, and Robert Mueller's crew investigating the Trump campaign h… https://t.co/PjZcl7KdhX
    {'neg': 0.0, 'neu': 0.825, 'pos': 0.175, 'compound': 0.5267}
    Mon Dec 11 04:14:01 +0000 2017
     
    On @WattersWorld, @PressSec Sarah Sanders slammed former President @BarackObama for taking credit for the economy u… https://t.co/y7ShRMmJ9q
    {'neg': 0.0, 'neu': 0.86, 'pos': 0.14, 'compound': 0.3818}
    Mon Dec 11 04:12:02 +0000 2017
     
    On @ffweekend, @GovMikeHuckabee responded to @ChelseaHandler, who tweeted a vulgar video that mocks @PressSec's wei… https://t.co/RhrGg1lYpn
    {'neg': 0.176, 'neu': 0.824, 'pos': 0.0, 'compound': -0.4588}
    Mon Dec 11 04:07:06 +0000 2017
     
    .@HeyTammyBruce: "Since [@POTUS] was elected, 2.2 million jobs were created... We want that to continue." https://t.co/u7d0vqFwAS
    {'neg': 0.0, 'neu': 0.92, 'pos': 0.08, 'compound': 0.0772}
    Mon Dec 11 04:05:01 +0000 2017
     
    .@PressSec: "It's laughable that President Obama thinks he has anything to do with the success of where the economy… https://t.co/kjoQyOpgNz
    {'neg': 0.0, 'neu': 0.786, 'pos': 0.214, 'compound': 0.5994}
    Mon Dec 11 04:00:04 +0000 2017
     
    On "Sunday Morning Futures," @RepPeteKing pushed back on those claiming @DonaldJTrumpJr engaged in collusion with R… https://t.co/u6MpfSs3Lr
    {'neg': 0.0, 'neu': 0.856, 'pos': 0.144, 'compound': 0.4019}
    Mon Dec 11 03:56:01 +0000 2017
     
    .@PeterRoskam: "We've got to figure out how to give more middle income tax relief." https://t.co/AYnn1kZ2Iq
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 03:55:05 +0000 2017
     
    .@SteveHiltonx: "Republicans are wrong to claim that their tax bill will raise incomes as well. They don't seem to… https://t.co/5ubvXJNRaa
    {'neg': 0.134, 'neu': 0.776, 'pos': 0.091, 'compound': -0.25}
    Mon Dec 11 03:39:57 +0000 2017
     
    'Queens of the Stone Age' singer apologizes for kicking photographer in his audience https://t.co/AXNq6eBbo5
    {'neg': 0.0, 'neu': 0.839, 'pos': 0.161, 'compound': 0.3612}
    Mon Dec 11 03:33:04 +0000 2017
     
    New evacuations as huge Southern California fire flares up https://t.co/SyZtnsbZcs
    {'neg': 0.189, 'neu': 0.63, 'pos': 0.181, 'compound': -0.0258}
    Mon Dec 11 03:27:57 +0000 2017
     
    Pelosi suggests Trump early days in office like neophyte performing 'brain surgery' https://t.co/fjDqHcBn7x
    {'neg': 0.0, 'neu': 0.828, 'pos': 0.172, 'compound': 0.3612}
    Mon Dec 11 03:18:31 +0000 2017
     
    Right whales facing extinction after 17 die this year, scientists say https://t.co/In8c0Co8zf
    {'neg': 0.262, 'neu': 0.738, 'pos': 0.0, 'compound': -0.5994}
    Mon Dec 11 03:18:06 +0000 2017
     
    One dead after small plane crashes near Miami airport https://t.co/aay8Kxw71u
    {'neg': 0.323, 'neu': 0.677, 'pos': 0.0, 'compound': -0.6486}
    Mon Dec 11 03:14:27 +0000 2017
     
    Earlier today, @FLOTUS called for charitable giving as the holidays approach. https://t.co/CIWZkTLmup
    {'neg': 0.0, 'neu': 0.539, 'pos': 0.461, 'compound': 0.7717}
    Mon Dec 11 03:07:01 +0000 2017
     
    Self-propelled vessel intercepted smuggling more than 3,800 pounds of cocaine near Texas https://t.co/IDFCWeVXel https://t.co/3JNQh23DLb
    {'neg': 0.193, 'neu': 0.807, 'pos': 0.0, 'compound': -0.4767}
    Mon Dec 11 03:06:24 +0000 2017
     
    A.B. Stoddard on @facebook Messenger for kids: “When you are on those sites you are presenting a fake version of yo… https://t.co/AinycBbIRD
    {'neg': 0.134, 'neu': 0.866, 'pos': 0.0, 'compound': -0.4767}
    Mon Dec 11 03:01:32 +0000 2017
     
    .@CLewandowski_: "It took @BarackObama until February of his second year to get the ObamaCare bill done and it look… https://t.co/QAQZuOWirk
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 02:56:14 +0000 2017
     
    .@CLewandowski_ on the GOP's future: "The people of this great country voted for change and that change is… https://t.co/babVRr8Rhv
    {'neg': 0.0, 'neu': 0.787, 'pos': 0.213, 'compound': 0.7073}
    Mon Dec 11 02:51:49 +0000 2017
     
    .@CLewandowski_: "If a Republican is not willing to play ball, [@POTUS] is willing to call them out to fulfill the… https://t.co/a1TGoxuZBl
    {'neg': 0.089, 'neu': 0.785, 'pos': 0.126, 'compound': 0.2177}
    Mon Dec 11 02:43:53 +0000 2017
     
    .@CLewandowski_: "@POTUS' base is stronger and more visceral today than it was on Election Day a year ago."… https://t.co/ASOdnpzel3
    {'neg': 0.0, 'neu': 0.867, 'pos': 0.133, 'compound': 0.3818}
    Mon Dec 11 02:39:22 +0000 2017
     
    .@CLewandowski_ on @POTUS' Israel decision: "There are 12 presidents preceding him that pledged to do what he did,… https://t.co/UmkaVWtpWr
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 02:35:57 +0000 2017
     
    .@TomiLahren on @chelseahandler: "She doesn't realize it, but she's @realDonaldTrump. She goes on Twitter, says som… https://t.co/3rNlCJDgSM
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 02:33:12 +0000 2017
     
    .@guypbenson: "It is exhausting to have politics shoved down my throat in every corner of American life all the tim… https://t.co/owx7gcPw9E
    {'neg': 0.111, 'neu': 0.889, 'pos': 0.0, 'compound': -0.3612}
    Mon Dec 11 02:28:48 +0000 2017
     
    .@TomiLahren on GOP tax plan: "It's sending the message that we're gonna do something in Washington DC." @NextRevFNC https://t.co/uS0aX04DRO
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 02:25:08 +0000 2017
     
    .@guypbensonon GOP tax plan: "While it may not raise wages, it will allow people to keep more of their own money ra… https://t.co/yoUxI14PNk
    {'neg': 0.0, 'neu': 0.921, 'pos': 0.079, 'compound': 0.2263}
    Mon Dec 11 02:20:08 +0000 2017
     
    .@SteveHiltonx: "Republicans are wrong to claim that their tax bill will raise incomes as well. They don't seem to… https://t.co/CN9RDW4k0n
    {'neg': 0.134, 'neu': 0.776, 'pos': 0.091, 'compound': -0.25}
    Mon Dec 11 02:18:18 +0000 2017
     
    Tigertown: Morris, Trammell elected to baseball Hall of Fame https://t.co/k4oeC4B79M
    {'neg': 0.0, 'neu': 0.756, 'pos': 0.244, 'compound': 0.4404}
    Mon Dec 11 01:52:56 +0000 2017
     
    Jerry Brown: Trump doesn't fear 'the wrath of God' https://t.co/Vf4DSUpfZl
    {'neg': 0.0, 'neu': 0.629, 'pos': 0.371, 'compound': 0.5759}
    Mon Dec 11 01:45:28 +0000 2017
     
    Democratic campaign flyer compares Roy Moore to George Wallace, says he supports segregation https://t.co/7ewJ1ob2gm
    {'neg': 0.0, 'neu': 0.839, 'pos': 0.161, 'compound': 0.3612}
    Mon Dec 11 01:35:35 +0000 2017
     
    Mike Huckabee: Chelsea Handler's 'Vicious Attacks' Are Beginning to Backfire https://t.co/8aVWEi3DuZ
    {'neg': 0.375, 'neu': 0.625, 'pos': 0.0, 'compound': -0.6597}
    Mon Dec 11 01:20:48 +0000 2017
     
    Goodell surprises terminally ill NY volunteer firefighter with Super Bowl tickets https://t.co/2VxzZSbKx0
    {'neg': 0.159, 'neu': 0.511, 'pos': 0.33, 'compound': 0.4588}
    Mon Dec 11 01:10:23 +0000 2017
     
    Armie Hammer apologizes for comments about Casey Affleck, sexual assault https://t.co/YsRDA3MtYl
    {'neg': 0.248, 'neu': 0.588, 'pos': 0.163, 'compound': -0.3182}
    Mon Dec 11 01:06:39 +0000 2017
     
    RIGHT NOW: Brian @Kilmeade Goes Inside the Life and Legacy of Andrew Jackson - Tune in at 8p &amp; 11p ET on Fox News C… https://t.co/xzmRjuuV8G
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 01:00:04 +0000 2017
     
    Pence responds to Abbas snub over Jerusalem decision https://t.co/qinBMGmKac
    {'neg': 0.259, 'neu': 0.741, 'pos': 0.0, 'compound': -0.4215}
    Mon Dec 11 00:59:58 +0000 2017
     
    'Coco' wins the box office again as theaters prepare for 'Star Wars: The Last Jedi' https://t.co/17yNivHLb7
    {'neg': 0.169, 'neu': 0.657, 'pos': 0.174, 'compound': 0.0258}
    Mon Dec 11 00:45:50 +0000 2017
     
    Rep. King: Worst Thing Trump Could Do Is Remove Mueller https://t.co/PkH1lj73CG
    {'neg': 0.291, 'neu': 0.709, 'pos': 0.0, 'compound': -0.6249}
    Mon Dec 11 00:38:23 +0000 2017
     
    RT @EricShawnTV: Not just Russia: A new call tonight for a 2nd Independent Counsel to investigate the #Obama Justice Department's handling…
    {'neg': 0.0, 'neu': 0.841, 'pos': 0.159, 'compound': 0.5267}
    Mon Dec 11 00:33:54 +0000 2017
     
    Woman who looked at eclipse suffered crescent-shaped eye damage, study shows https://t.co/OHsZnVVkmv
    {'neg': 0.39, 'neu': 0.61, 'pos': 0.0, 'compound': -0.7506}
    Mon Dec 11 00:26:43 +0000 2017
     
    OPINION: North Korea won't start a war - @realDonaldTrump shouldn't launch an attack https://t.co/PkqBVrsoDj
    {'neg': 0.0, 'neu': 0.637, 'pos': 0.363, 'compound': 0.6908}
    Mon Dec 11 00:21:55 +0000 2017
     
    OPINION: Obama still doesn’t get it – optimism, not whining, grows an economy https://t.co/wvdPYwYMLp
    {'neg': 0.0, 'neu': 0.68, 'pos': 0.32, 'compound': 0.6329}
    Mon Dec 11 00:14:39 +0000 2017
     
    Jonathan Wachtel on @POTUS' Israel decision: "Of course there's a lot of concern in the world about where all this… https://t.co/U68BNEXiW2
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 00:10:21 +0000 2017
     
    Jonathan Wachtel on @POTUS' Israel decision: "Look at that speech word by word and you will see plenty of room in t… https://t.co/HA7laZSQVo
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 23:58:40 +0000 2017
     
    Jonathan Wachtel on @POTUS' Israel decision: "President #Trump isn't the only US president to have taken a bold ste… https://t.co/I6XgbQOMkr
    {'neg': 0.0, 'neu': 0.874, 'pos': 0.126, 'compound': 0.3818}
    Sun Dec 10 23:56:34 +0000 2017
     
    .@RepDougCollins on @HillaryClinton investigation: "[Clinton] has been around this town, she was not unaware of wha… https://t.co/q8jgNevL4t
    {'neg': 0.0, 'neu': 0.91, 'pos': 0.09, 'compound': 0.1511}
    Sun Dec 10 23:46:50 +0000 2017
     
    Hundreds of homes destroyed in California wildfires, @WillCarrFNC reports. @ANHQDC https://t.co/D1aSjJ5SWz https://t.co/a4a42SHdGy
    {'neg': 0.225, 'neu': 0.775, 'pos': 0.0, 'compound': -0.4939}
    Sun Dec 10 23:35:48 +0000 2017
     
    'Stain on America!' @realDonaldTrump denounces 'Fake News Media' after string of major reporting errors exposed https://t.co/lcX2qh2WuV
    {'neg': 0.454, 'neu': 0.546, 'pos': 0.0, 'compound': -0.8398}
    Sun Dec 10 23:34:08 +0000 2017
     
    Roy Moore denies misconduct allegations in new interview: 'I did not date underaged women' https://t.co/QiGmosf6eG
    {'neg': 0.167, 'neu': 0.833, 'pos': 0.0, 'compound': -0.4215}
    Sun Dec 10 23:16:01 +0000 2017
     
    North Korea is hacking bitcoin exchanges as currency value soars, expert says https://t.co/jBDuJAYBjG
    {'neg': 0.0, 'neu': 0.833, 'pos': 0.167, 'compound': 0.34}
    Sun Dec 10 23:11:03 +0000 2017
     
    TONIGHT: Brian @Kilmeade Goes Inside the Life and Legacy of Andrew Jackson - Tune in at 8p &amp; 11p ET on Fox News Cha… https://t.co/vK2B5y9nBz
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 23:07:03 +0000 2017
     
    Pelosi suggests Trump early days in office like neophyte performing 'brain surgery' https://t.co/fjDqHcSYw7
    {'neg': 0.0, 'neu': 0.828, 'pos': 0.172, 'compound': 0.3612}
    Sun Dec 10 23:02:01 +0000 2017
     
    At his rally Friday in Florida, @POTUS reiterated his hard-line stance on immigration. https://t.co/23mOEFXJ7a https://t.co/lTWYHgjgJS
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:56:03 +0000 2017
     
    .@JudgeJeanine: "There have been times in our history where corruption and lawlessness were so pervasive, that exam… https://t.co/ch5iDTkxxW
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:51:04 +0000 2017
     
    Intruder shot in Pennsylvania home invasion attempt was homeowner's relative, police say https://t.co/GLQCZ1cKBV
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:50:34 +0000 2017
     
    #ALSen Poll Average: Moore leads Jones 49.1% to 45.3%. https://t.co/1EIM8OFjvy
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:46:37 +0000 2017
     
    .@GOPLeader: "I have never watched a President so engaged... He wants to keep his promise to create more jobs and m… https://t.co/RU4nIm3bbK
    {'neg': 0.0, 'neu': 0.812, 'pos': 0.188, 'compound': 0.5267}
    Sun Dec 10 22:45:26 +0000 2017
     
    Do you agree with @JudgeJeanine? https://t.co/gllDDJFkna https://t.co/eCZmbrOiET
    {'neg': 0.0, 'neu': 0.706, 'pos': 0.294, 'compound': 0.3612}
    Sun Dec 10 22:42:05 +0000 2017
     
    'Nancy Pelosi Is a Stooge': Bossie Reacts After House Minority Leader Criticizes Trump https://t.co/mIR88peLdE
    {'neg': 0.167, 'neu': 0.833, 'pos': 0.0, 'compound': -0.34}
    Sun Dec 10 22:40:26 +0000 2017
     
    Haley: Trump's Jerusalem decision will 'fastball' peace, stop Israel 'bashing' https://t.co/xvsPJ0HS4D
    {'neg': 0.15, 'neu': 0.612, 'pos': 0.238, 'compound': 0.3182}
    Sun Dec 10 22:40:07 +0000 2017
     
    RT @EricShawnTV: I anchor at 6pm ET @FoxNews: The @POTUS @realDonaldTrump Peace plan for #Israel and the #Palestinians. Can he succeed? Her…
    {'neg': 0.0, 'neu': 0.739, 'pos': 0.261, 'compound': 0.7717}
    Sun Dec 10 22:39:26 +0000 2017
     
    Netanyahu slams Turkey's Erdogan for claiming Israel is 'terrorist state' that 'kills children' https://t.co/GH4kgtHMOX
    {'neg': 0.406, 'neu': 0.594, 'pos': 0.0, 'compound': -0.8481}
    Sun Dec 10 22:37:34 +0000 2017
     
    .@PressSec: "It's laughable that President Obama thinks he has anything to do with the success of where the economy… https://t.co/0kFm8uaokf
    {'neg': 0.0, 'neu': 0.786, 'pos': 0.214, 'compound': 0.5994}
    Sun Dec 10 22:36:02 +0000 2017
     
    Mike Eruzione, 1980 U.S. Men's Hockey Captain: "[@POTUS] is somebody that you should respect. Whether you agree or… https://t.co/PsJTcCcjON
    {'neg': 0.0, 'neu': 0.752, 'pos': 0.248, 'compound': 0.6808}
    Sun Dec 10 22:30:02 +0000 2017
     
    .@HeyTammyBruce: "Since [@POTUS] was elected, 2.2 million jobs were created... We want that to continue." https://t.co/lLHFVOxS3I
    {'neg': 0.0, 'neu': 0.92, 'pos': 0.08, 'compound': 0.0772}
    Sun Dec 10 22:28:43 +0000 2017
     
    Catholic priest tells black family he was 'blinded by hate,' apologizes for burning cross 40 years ago on their law… https://t.co/yYp2wOXPum
    {'neg': 0.0, 'neu': 0.889, 'pos': 0.111, 'compound': 0.3612}
    Sun Dec 10 22:27:52 +0000 2017
     
    Do you agree with @JudgeJeanine? https://t.co/gllDDJWVLK https://t.co/n5U8fvV8OP
    {'neg': 0.0, 'neu': 0.706, 'pos': 0.294, 'compound': 0.3612}
    Sun Dec 10 22:21:16 +0000 2017
     
    Presidents @BillClinton, George W. Bush, and @BarackObama all declared Jerusalem the capital of Israel. https://t.co/Ah7klpcFFy
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:17:03 +0000 2017
     
    Rutgers announces punishment for professor accused of making anti-Semitic remarks https://t.co/DEfcQtRVpG
    {'neg': 0.375, 'neu': 0.625, 'pos': 0.0, 'compound': -0.6597}
    Sun Dec 10 22:09:50 +0000 2017
     
    The San Diego Fire-Rescue Department suited up Emily the arson investigation dog amid concern she was inhaling smok… https://t.co/StVyPJ6lQr
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 21:59:29 +0000 2017
     
    .@WWE wrestler Rich Swann arrested, charged with battery of wrestler wife. https://t.co/9sNI5GCsUR
    {'neg': 0.28, 'neu': 0.514, 'pos': 0.206, 'compound': -0.0772}
    Sun Dec 10 21:54:07 +0000 2017
     
    On @ffweekend, @SheriffClarke slammed Rep. John Lewis for boycotting @POTUS's appearance at a Civil Rights museum i… https://t.co/ThZPN1yHx5
    {'neg': 0.144, 'neu': 0.856, 'pos': 0.0, 'compound': -0.4019}
    Sun Dec 10 21:44:06 +0000 2017
     
    Boxer Stephen Smith's ear almost ripped off in Vegas bout https://t.co/ZyyeRusMD8
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 21:37:42 +0000 2017
     
    Haley: Trump's Jerusalem decision will 'fastball' peace, stop Israel 'bashing' https://t.co/xvsPJ0HS4D
    {'neg': 0.15, 'neu': 0.612, 'pos': 0.238, 'compound': 0.3182}
    Sun Dec 10 21:31:29 +0000 2017
     
    Princes William and Harry pick sculptor for Princess Diana statue https://t.co/TQ7F6Mso9F
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 21:24:21 +0000 2017
     
    #ALSen Poll Average: Moore leads Jones 49.1% to 45.3%. https://t.co/6Y2a2VAfBf
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 21:14:05 +0000 2017
     
    Sarah Sanders: 'Laughable' for Obama to Try to Take Credit for Economy https://t.co/BxoqvvIAM8
    {'neg': 0.0, 'neu': 0.822, 'pos': 0.178, 'compound': 0.3818}
    Sun Dec 10 21:09:11 +0000 2017
     
    Comedian @hannibalburess arrested for disorderly intoxication https://t.co/oAqQxCQMKU
    {'neg': 0.29, 'neu': 0.467, 'pos': 0.243, 'compound': -0.128}
    Sun Dec 10 20:59:00 +0000 2017
     
    .@Wendys trolls fast food hack on Twitter https://t.co/Ti1zUMjJ3y
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:52:51 +0000 2017
     
    .@GovMikeHuckabee: "[@ChelseaHandler's] called a late night host. I think it's so late now that she's not on anymor… https://t.co/z0LKHSdXqr
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:51:04 +0000 2017
     
    .@nikkihaley: Jerusalem Decision Will Speed Up Middle East Peace Process https://t.co/GvI110MnKT
    {'neg': 0.0, 'neu': 0.741, 'pos': 0.259, 'compound': 0.5423}
    Sun Dec 10 20:48:37 +0000 2017
     
    Hillary Clinton portrait alerts security dogs, causes road closures in Miami https://t.co/crgFOpcXCM
    {'neg': 0.0, 'neu': 0.821, 'pos': 0.179, 'compound': 0.34}
    Sun Dec 10 20:43:54 +0000 2017
     
    .@JudgeJeanine: "There is a cleansing needed in our FBI and Department of Justice. It needs to be cleansed of indiv… https://t.co/EQixmUybUe
    {'neg': 0.0, 'neu': 0.848, 'pos': 0.152, 'compound': 0.5267}
    Sun Dec 10 20:32:01 +0000 2017
     
    Comstock opposes GOP support for Moore, suggests he'll face Hill ethics probe https://t.co/uxLqnZwwNt
    {'neg': 0.0, 'neu': 0.816, 'pos': 0.184, 'compound': 0.4019}
    Sun Dec 10 20:31:19 +0000 2017
     
    .@NancyPelosi suggests @POTUS early days in office like neophyte performing 'brain surgery' https://t.co/WLfeOGxAOs https://t.co/UIfS8LqtL2
    {'neg': 0.0, 'neu': 0.839, 'pos': 0.161, 'compound': 0.3612}
    Sun Dec 10 20:25:05 +0000 2017
     
    WATCH Part 2 of @nikkihaley's interview about @POTUS' Jerusalem decision and North Korea sanctions on… https://t.co/vxHSb2PmTk
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:18:06 +0000 2017
     
    .@SheriffClarke: "If it weren't for the Republicans in Congress in the 1960s, there would be no Civil Rights Act. T… https://t.co/oh6PowT7Nn
    {'neg': 0.099, 'neu': 0.901, 'pos': 0.0, 'compound': -0.296}
    Sun Dec 10 20:18:02 +0000 2017
     
    On @ffweekend, @SheriffClarke slammed Rep. John Lewis for boycotting @POTUS's appearance at a Civil Rights museum i… https://t.co/rzkhptkwKc
    {'neg': 0.144, 'neu': 0.856, 'pos': 0.0, 'compound': -0.4019}
    Sun Dec 10 20:17:02 +0000 2017
     
    WATCH Part 1 of @nikkihaley's interview about @POTUS' Jerusalem decision and North Korea sanctions on… https://t.co/4ywwlcXajG
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:10:27 +0000 2017
     
    TONIGHT: Brian @Kilmeade Goes Inside the Life and Legacy of Andrew Jackson - Tune in at 8p &amp; 11p ET on Fox News Cha… https://t.co/MVXkKU7bA0
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:07:01 +0000 2017
     
    Swiss woman angry over champagne refusal ejected from plane 
    
    https://t.co/DsVESiiYWO
    {'neg': 0.244, 'neu': 0.593, 'pos': 0.163, 'compound': -0.2732}
    Sun Dec 10 20:06:39 +0000 2017
     
    RT @EricShawnTV: I anchor @FoxNews at 4 &amp; 6 pm ET: "If North Korea even attempts to try and threaten the United States or any one of our al…
    {'neg': 0.088, 'neu': 0.816, 'pos': 0.095, 'compound': 0.0516}
    Sun Dec 10 20:05:30 +0000 2017
     
    Moments ago, @FLOTUS called for charitable giving as the holidays approach. https://t.co/bMqn6mBKMs
    {'neg': 0.0, 'neu': 0.539, 'pos': 0.461, 'compound': 0.7717}
    Sun Dec 10 20:01:49 +0000 2017
     
    .@nikkihaley on North Korea sanctions: "The last time [China] completely cut off the oil, North Korea came to the t… https://t.co/KddIYS43iP
    {'neg': 0.107, 'neu': 0.893, 'pos': 0.0, 'compound': -0.3384}
    Sun Dec 10 19:58:07 +0000 2017
     
    .@nikkihaley on Jerusalem decision: "When you bully Israel, you are not helping the peace process. We see Israel as… https://t.co/pxZtiiahOq
    {'neg': 0.318, 'neu': 0.682, 'pos': 0.0, 'compound': -0.7869}
    Sun Dec 10 19:51:19 +0000 2017
     
    California wildfires kill at least 1 as high winds threaten more outbreaks 
    
    https://t.co/9rLHJ0OZFP
    {'neg': 0.422, 'neu': 0.578, 'pos': 0.0, 'compound': -0.8074}
    Sun Dec 10 19:50:25 +0000 2017
     
    On @ffweekend, @Varneyco praised @POTUS's handling of the economy. https://t.co/0QufAxXL4E
    {'neg': 0.0, 'neu': 0.738, 'pos': 0.262, 'compound': 0.4939}
    Sun Dec 10 19:49:06 +0000 2017
     
    .@nikkihaley on Jerusalem decision: "We will respect anything that the two parties come together on."… https://t.co/K5Mpz3e8nh
    {'neg': 0.0, 'neu': 0.707, 'pos': 0.293, 'compound': 0.7003}
    Sun Dec 10 19:47:43 +0000 2017
     
    .@nikkihaley on Jerusalem decision: "[@POTUS] strongly believes that those final statuses should be decided between… https://t.co/Ib5Hi7UCF1
    {'neg': 0.0, 'neu': 0.877, 'pos': 0.123, 'compound': 0.2732}
    Sun Dec 10 19:45:55 +0000 2017
     
    .@nikkihaley on Jerusalem decision: "We didn't set any parameters, we didn't say this was the final status, what we… https://t.co/Iwdk8B9tvQ
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 19:43:18 +0000 2017
     



```python
fox_complete_df = pd.DataFrame({'Date (Fox)': fox_date_list, 'Tweet (Fox)': fox_text_list, 
                                'Negative Score (Fox)': fox_negative_list, 'Neutral Score (Fox)': fox_neutral_list, 
                                'Positive Score (Fox)': fox_positive_list, 'Compound Score (Fox)': fox_compound})
fox_final_df = fox_complete_df[['Date (Fox)', 'Tweet (Fox)', 'Negative Score (Fox)', 'Neutral Score (Fox)', 
                 'Positive Score (Fox)', 'Compound Score (Fox)']]
fox_final_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date (Fox)</th>
      <th>Tweet (Fox)</th>
      <th>Negative Score (Fox)</th>
      <th>Neutral Score (Fox)</th>
      <th>Positive Score (Fox)</th>
      <th>Compound Score (Fox)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mon Dec 11 04:17:05 +0000 2017</td>
      <td>On @ffweekend, @SheriffClarke slammed Rep. Joh...</td>
      <td>0.144</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>-0.4019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mon Dec 11 04:15:02 +0000 2017</td>
      <td>In his speech at the opening of the Mississipp...</td>
      <td>0.000</td>
      <td>0.826</td>
      <td>0.174</td>
      <td>0.6369</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon Dec 11 04:14:01 +0000 2017</td>
      <td>.@JesseBWatters: "The @FBI, the Department of ...</td>
      <td>0.000</td>
      <td>0.825</td>
      <td>0.175</td>
      <td>0.5267</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mon Dec 11 04:12:02 +0000 2017</td>
      <td>On @WattersWorld, @PressSec Sarah Sanders slam...</td>
      <td>0.000</td>
      <td>0.860</td>
      <td>0.140</td>
      <td>0.3818</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mon Dec 11 04:07:06 +0000 2017</td>
      <td>On @ffweekend, @GovMikeHuckabee responded to @...</td>
      <td>0.176</td>
      <td>0.824</td>
      <td>0.000</td>
      <td>-0.4588</td>
    </tr>
  </tbody>
</table>
</div>




```python
fox_compound_df = pd.DataFrame(fox_compound)
fox_compound_df.columns = ['Fox Compound Score']
fox_compound_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fox Compound Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.4019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.6369</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.5267</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.3818</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.4588</td>
    </tr>
  </tbody>
</table>
</div>




```python
#New York Times

nyt_text_list = []
nyt_date_list = []
nyt_negative_list = []
nyt_neutral_list = []
nyt_positive_list = []
nyt_compound = []


nyt_public_tweets = api.user_timeline(nyt, count = 100)

for nyt_tweet in nyt_public_tweets:
    
    nyt_text = nyt_tweet['text']
    nyt_date = nyt_tweet['created_at']
    
    print(nyt_text)
    
    
    nyt_scores = analyzer.polarity_scores(nyt_text)
    print(nyt_scores)
    print(nyt_date)
    print(' ')
    
    nyt_text_list.append(nyt_text)
    nyt_date_list.append(nyt_date)
    nyt_negative_list.append(nyt_scores['neg'])
    nyt_neutral_list.append(nyt_scores['neu'])
    nyt_positive_list.append(nyt_scores['pos'])
    nyt_compound.append(nyt_scores['compound'])
```

    Your cat tattoo can have your actual cat in it https://t.co/wyrGCMsrld
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 04:12:09 +0000 2017
     
    RT @nytimesworld: Protests in Lebanon near U.S. embassy after Trump’s Jerusalem decision https://t.co/6QjgtAgUVj https://t.co/MVThAPnypJ
    {'neg': 0.128, 'neu': 0.872, 'pos': 0.0, 'compound': -0.2263}
    Mon Dec 11 03:53:03 +0000 2017
     
    Horror films dominated the cultural conversation in 2017. So @nytmag asked the year's best actors to perform in a s… https://t.co/1zY2LD0VYi
    {'neg': 0.143, 'neu': 0.695, 'pos': 0.162, 'compound': 0.128}
    Mon Dec 11 03:39:09 +0000 2017
     
    RT @nytimesworld: London's Heathrow Airport was brought to a virtual halt on Sunday after a snowstorm swept through the region https://t.co…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 03:25:04 +0000 2017
     
    17 ways to celebrate the season in New York City, from the familiar (Rockettes!) to the far-out (Taylor Mac!) https://t.co/50cu9mc8bC
    {'neg': 0.0, 'neu': 0.816, 'pos': 0.184, 'compound': 0.6467}
    Mon Dec 11 03:13:26 +0000 2017
     
    9 ways to work better in 2018 https://t.co/LMKfNrcMVS
    {'neg': 0.0, 'neu': 0.674, 'pos': 0.326, 'compound': 0.4404}
    Mon Dec 11 02:56:51 +0000 2017
     
    A high-powered attorney died. His ex-wife investigated and found a web of drug abuse in his profession.… https://t.co/U9UP9O0Ryt
    {'neg': 0.358, 'neu': 0.642, 'pos': 0.0, 'compound': -0.8316}
    Mon Dec 11 02:40:03 +0000 2017
     
    Op-Ed Columnist: Susan Collins and the Duping of Centrists https://t.co/sGjJ28NHxJ
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 02:16:54 +0000 2017
     
    In a Blow to Hamas, Israel Destroys Tunnel From Gaza https://t.co/qvmAm1MCIY
    {'neg': 0.286, 'neu': 0.714, 'pos': 0.0, 'compound': -0.5574}
    Mon Dec 11 02:00:51 +0000 2017
     
    "Dark" plays like a slower, artsier, more complicated take on "Stranger Things," our critic writes https://t.co/mwFkGjnwjd
    {'neg': 0.113, 'neu': 0.645, 'pos': 0.242, 'compound': 0.34}
    Mon Dec 11 01:52:46 +0000 2017
     
    2017: The Year in New York https://t.co/oMSRlih8ib
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 01:35:34 +0000 2017
     
    President Trump called for a Washington Post reporter to be fired over a misleading tweet https://t.co/NLAYpP1rDI
    {'neg': 0.344, 'neu': 0.656, 'pos': 0.0, 'compound': -0.743}
    Mon Dec 11 01:17:47 +0000 2017
     
    Melting glaciers, wicked hot summers, water crises. These are the climate stories to read from this year. https://t.co/6tmy4MU3z0
    {'neg': 0.167, 'neu': 0.833, 'pos': 0.0, 'compound': -0.5267}
    Mon Dec 11 00:57:34 +0000 2017
     
    A dying man had “do not resuscitate” tattooed on his chest. Doctors said it “produced more confusion than clarity.” https://t.co/d1C8KThhvB
    {'neg': 0.122, 'neu': 0.878, 'pos': 0.0, 'compound': -0.3597}
    Mon Dec 11 00:40:33 +0000 2017
     
    The Look: After-School Special https://t.co/MDkn8XkFiy
    {'neg': 0.0, 'neu': 0.597, 'pos': 0.403, 'compound': 0.4019}
    Mon Dec 11 00:22:04 +0000 2017
     
    The risk can’t be ignored but isn’t as great as it may seem from a recent study https://t.co/VYh6cV0i4H
    {'neg': 0.14, 'neu': 0.613, 'pos': 0.247, 'compound': 0.6652}
    Mon Dec 11 00:17:42 +0000 2017
     
    Op-Ed Contributor: The Man Who Danced on the Heads of Snakes https://t.co/WBlhVk3eLq
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Mon Dec 11 00:01:25 +0000 2017
     
    The American ambassador to the UN said that the women who have accused Trump of sexual misconduct “should be heard” https://t.co/tgad1uL3Qp
    {'neg': 0.099, 'neu': 0.901, 'pos': 0.0, 'compound': -0.296}
    Sun Dec 10 23:47:55 +0000 2017
     
    Luke Skywalker’s lineage is a narrative backbone in "Star Wars." So it's natural to wonder: who are Rey's parents?… https://t.co/w9Mfts2iaA
    {'neg': 0.0, 'neu': 0.866, 'pos': 0.134, 'compound': 0.4173}
    Sun Dec 10 23:30:13 +0000 2017
     
    How ISIS Produced Its Cruel Arsenal on an Industrial Scale https://t.co/afprGpE6NB
    {'neg': 0.275, 'neu': 0.725, 'pos': 0.0, 'compound': -0.5859}
    Sun Dec 10 23:21:37 +0000 2017
     
    RT @nytopinion: Over the last 50 years or so, the rules about a woman’s place were shattered. I saw all this happen, and it knocks me out w…
    {'neg': 0.11, 'neu': 0.89, 'pos': 0.0, 'compound': -0.4767}
    Sun Dec 10 23:10:01 +0000 2017
     
    Porgs, Ewoks and Rathtars: A Field Guide to ‘Star Wars’ Creatures https://t.co/S6siGWTZYo
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 23:02:21 +0000 2017
     
    Boris Johnson leaves Tehran without a clear statement on imprisoned Brits and a tank dispute https://t.co/M9iwZQJbF4
    {'neg': 0.417, 'neu': 0.583, 'pos': 0.0, 'compound': -0.7835}
    Sun Dec 10 23:00:26 +0000 2017
     
    An NYT reader's reaction to an article about President Trump’s daily fight for self-preservation… https://t.co/F705vf9Qoi
    {'neg': 0.157, 'neu': 0.843, 'pos': 0.0, 'compound': -0.3818}
    Sun Dec 10 22:50:38 +0000 2017
     
    “The Disaster Artist” had a very strong weekend, taking in $6.4 million in relatively limited release https://t.co/aqQ9xFGouJ
    {'neg': 0.266, 'neu': 0.575, 'pos': 0.159, 'compound': -0.3415}
    Sun Dec 10 22:45:05 +0000 2017
     
    Louis C.K. will buy back the rights to his film after sexual misconduct allegations led to the movie’s shelving https://t.co/F92nFIGdzZ
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:30:05 +0000 2017
     
    This "Odyssey" strips away formulaic language to let the characters take center stage https://t.co/EMx7fd4Q8S
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 22:15:07 +0000 2017
     
    Sometimes it’s hard to keep track of what’s arriving when. But we’re here to help. https://t.co/peCXk6zyrC
    {'neg': 0.077, 'neu': 0.773, 'pos': 0.149, 'compound': 0.3182}
    Sun Dec 10 22:00:08 +0000 2017
     
    Alexandra Bell puts The New York Times under the microscope, and offers her own “counternarratives” of the news… https://t.co/kPDfkmDx1g
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 21:45:07 +0000 2017
     
    “We were all born to die.” In our video, we follow a mother who fights her own personal demons as her son fights ca… https://t.co/s69owvxKdo
    {'neg': 0.197, 'neu': 0.803, 'pos': 0.0, 'compound': -0.6597}
    Sun Dec 10 21:30:14 +0000 2017
     
    A tour of Los Angeles’s flourishing pot scene https://t.co/lzSpDtPJkq
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 21:15:02 +0000 2017
     
    In a live TV remake of the holiday classic "A Christmas Story," Jane Krakowski plays Ralphie's teacher https://t.co/r1s6lxmM05
    {'neg': 0.0, 'neu': 0.761, 'pos': 0.239, 'compound': 0.5719}
    Sun Dec 10 21:00:07 +0000 2017
     
    As Alabama's Senate race winds down, residents are lamenting that policy issues didn't seize the spotlight https://t.co/5RaJtzyHAg
    {'neg': 0.158, 'neu': 0.842, 'pos': 0.0, 'compound': -0.4588}
    Sun Dec 10 20:45:13 +0000 2017
     
    The week’s top stories, and a look ahead https://t.co/zoAsvlH7W5
    {'neg': 0.0, 'neu': 0.795, 'pos': 0.205, 'compound': 0.2023}
    Sun Dec 10 20:30:06 +0000 2017
     
    RT @danielle_ivory: NEW: Confidential internal EPA documents obtained by the @nytimes show that a slowdown of enforcement coincides with po…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 20:23:00 +0000 2017
     
    RT @nytimesbooks: Did you love "Brown Girl Dreaming"? @JackieWoodson will publish 2 new books for young readers next year. https://t.co/9jj…
    {'neg': 0.0, 'neu': 0.811, 'pos': 0.189, 'compound': 0.6369}
    Sun Dec 10 20:19:26 +0000 2017
     
    Watching the "Me, Too" campaign, some particularly courageous Afghan women are more willing to talk about assaults https://t.co/Ws52CUJm2o
    {'neg': 0.151, 'neu': 0.69, 'pos': 0.159, 'compound': 0.0498}
    Sun Dec 10 20:15:02 +0000 2017
     
    He's 22, she's 81, and their friendship is melting hearts. A forgotten war orphan. The weird pot scene in L.A. https://t.co/epDpoj7vEX
    {'neg': 0.284, 'neu': 0.606, 'pos': 0.11, 'compound': -0.5574}
    Sun Dec 10 20:00:08 +0000 2017
     
    The risk can’t be ignored but isn’t as great as it may seem from a recent study https://t.co/M0Gfsl6tft
    {'neg': 0.14, 'neu': 0.613, 'pos': 0.247, 'compound': 0.6652}
    Sun Dec 10 19:45:06 +0000 2017
     
    The American ambassador to the UN said that the women who have accused Trump of sexual misconduct “should be heard” https://t.co/eN6Wg9hVRo
    {'neg': 0.099, 'neu': 0.901, 'pos': 0.0, 'compound': -0.296}
    Sun Dec 10 19:38:47 +0000 2017
     
    “There is no place for anti-Semitism in our Swedish society. The perpetrators will answer for their crimes.” https://t.co/4MdjtivD9Z
    {'neg': 0.208, 'neu': 0.792, 'pos': 0.0, 'compound': -0.4939}
    Sun Dec 10 19:30:30 +0000 2017
     
    Pick up the phone rather than relying on apps https://t.co/j2WY3UJNOs
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 19:15:12 +0000 2017
     
    The map was expected to sell for over $1 million but was withdrawn from a Christie's sale https://t.co/yfDF7nfIXE
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 19:00:20 +0000 2017
     
    Who are Rey's parents? The internet has theories. https://t.co/YnIOnS6jJX https://t.co/U4qIOOS7Rt
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 18:45:16 +0000 2017
     
    Watch: Ashley Iorio and David Sands met at the United States Naval Academy and were wed on Veterans Day #Daily360… https://t.co/zsM8BN4HxV
    {'neg': 0.0, 'neu': 0.877, 'pos': 0.123, 'compound': 0.4215}
    Sun Dec 10 18:30:10 +0000 2017
     
    In an improbable turn of events, the sound of silence went viral https://t.co/oLYdOmma20
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 18:15:05 +0000 2017
     
    How to know if an app is appropriate for children, or even educational https://t.co/4LqtR4iPFg
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 18:00:17 +0000 2017
     
    At center stage of what looked like a Nine Inch Nails concert was Taylor Swift https://t.co/ByBFKKZWg1
    {'neg': 0.0, 'neu': 0.751, 'pos': 0.249, 'compound': 0.5106}
    Sun Dec 10 17:45:10 +0000 2017
     
    Senator Richard Shelby, a fixture of Republican politics in Alabama, said on national television that the party cou… https://t.co/zJePeEuBgO
    {'neg': 0.0, 'neu': 0.863, 'pos': 0.137, 'compound': 0.4019}
    Sun Dec 10 17:30:11 +0000 2017
     
    Opinion: It's been brutal for anyone who believes America’s public lands are best left in their natural state https://t.co/LzyFFs4u9h
    {'neg': 0.153, 'neu': 0.597, 'pos': 0.25, 'compound': 0.3818}
    Sun Dec 10 17:15:15 +0000 2017
     
    We’re interested in hearing from parents whose children faced a serious illness. We invite you to share your experi… https://t.co/U9UaNOtkNf
    {'neg': 0.163, 'neu': 0.571, 'pos': 0.265, 'compound': 0.3612}
    Sun Dec 10 17:02:48 +0000 2017
     
    “They gave me 50/50, but we’ve changed it to 51/49, so that I have the upper hand." https://t.co/WpJn4eejKV https://t.co/VZqrtaXUDY
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 17:00:21 +0000 2017
     
    President Trump called for a Washington Post reporter to be fired over a misleading tweet https://t.co/60rquXrO18
    {'neg': 0.344, 'neu': 0.656, 'pos': 0.0, 'compound': -0.743}
    Sun Dec 10 16:45:05 +0000 2017
     
    Thousands of protesters chanted slogans against Trump’s decision to recognize Jerusalem as the capital of Israel https://t.co/icMzSfgJ0k
    {'neg': 0.106, 'neu': 0.894, 'pos': 0.0, 'compound': -0.2263}
    Sun Dec 10 16:30:08 +0000 2017
     
    Rising incomes allow China’s retirees to enjoy a retirement that would have been unthinkable for their parents https://t.co/w4gBshsfmM
    {'neg': 0.0, 'neu': 0.746, 'pos': 0.254, 'compound': 0.6249}
    Sun Dec 10 16:15:07 +0000 2017
     
    RT @EricLiptonNYT: There's widespread evidence that the Trump admin has been rolling back federal regulations. How about enforcement target…
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 16:14:28 +0000 2017
     
    @lucasgrindley We lost that challenge. Sorry about the error. Thanks for catching it! https://t.co/sZeNajXoe4
    {'neg': 0.333, 'neu': 0.455, 'pos': 0.212, 'compound': -0.3382}
    Sun Dec 10 16:01:27 +0000 2017
     
    The EPA under President Trump has slowed down efforts to punish polluters https://t.co/R1s7qP5oTR
    {'neg': 0.368, 'neu': 0.632, 'pos': 0.0, 'compound': -0.7506}
    Sun Dec 10 16:00:05 +0000 2017
     
    @AnonyMs_One Thanks for alerting us to the error. Sorry about that! https://t.co/sZeNajXoe4
    {'neg': 0.265, 'neu': 0.556, 'pos': 0.179, 'compound': -0.1007}
    Sun Dec 10 15:57:05 +0000 2017
     
    Modern Love: What 9 years of therapy and "open" Scrabble taught her about love https://t.co/qAJC03ajuk (Correction:… https://t.co/SepmMvhYqT
    {'neg': 0.0, 'neu': 0.625, 'pos': 0.375, 'compound': 0.8555}
    Sun Dec 10 15:45:12 +0000 2017
     
    Max Clifford was serving an 8-year sentence for sexual offenses against victims as young as 15 https://t.co/XqLWWte10z
    {'neg': 0.239, 'neu': 0.697, 'pos': 0.065, 'compound': -0.5423}
    Sun Dec 10 15:30:03 +0000 2017
     
    Your cat tattoo can have your actual cat in it https://t.co/4pxWo0GJ4F
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 15:15:10 +0000 2017
     
    We interviewed members of 7 families who have spoken to parents, grandparents, sons or daughters about sexual assau… https://t.co/RGe2KNXXEI
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 15:00:16 +0000 2017
     
    In most places, a dollar is a dollar. But in the tax plan, the amount you make may be less important than how you m… https://t.co/Zx7pr7K69P
    {'neg': 0.0, 'neu': 0.939, 'pos': 0.061, 'compound': 0.1298}
    Sun Dec 10 14:30:24 +0000 2017
     
    Ojai is a picturesque town known for healthy, spiritual living. The California fires have disrupted that. https://t.co/bPYRgMz3es
    {'neg': 0.0, 'neu': 0.725, 'pos': 0.275, 'compound': 0.6486}
    Sun Dec 10 14:00:20 +0000 2017
     
    Some consumers are finding they can get a better deal on prescription drugs by leaving their insurance cards at home https://t.co/qK3Wrewxyp
    {'neg': 0.0, 'neu': 0.868, 'pos': 0.132, 'compound': 0.4404}
    Sun Dec 10 13:30:09 +0000 2017
     
    The Stone: For Veterans, a Path to Healing ‘Moral Injury’ https://t.co/69S0AXL1cF
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 13:07:39 +0000 2017
     
    Trump sees the presidency as a prize he must fight to protect every waking moment, and Twitter is his Excalibur… https://t.co/6aEvx25mLU
    {'neg': 0.102, 'neu': 0.667, 'pos': 0.231, 'compound': 0.5106}
    Sun Dec 10 13:00:16 +0000 2017
     
    Jerusalem is not just its postcard vistas. The day-in, day-out friction can be draining. https://t.co/kQqIbm7cqC
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 12:30:13 +0000 2017
     
    “Please do not shoot me": Newly released body camera footage shows a police officer shooting an unarmed man in Ariz… https://t.co/gN14S1iQu1
    {'neg': 0.0, 'neu': 0.903, 'pos': 0.097, 'compound': 0.2584}
    Sun Dec 10 12:00:04 +0000 2017
     
    Protests in Lebanon Near U.S. Embassy After Trump’s Jerusalem Decision https://t.co/rhTl1as2l0
    {'neg': 0.16, 'neu': 0.84, 'pos': 0.0, 'compound': -0.2263}
    Sun Dec 10 11:47:06 +0000 2017
     
    92 Somali citizens were flown out of the U.S. under orders of deportation, but their plane never made it to Somalia https://t.co/o0dc6Jm9A2
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 11:30:07 +0000 2017
     
    What Trump’s decision to recognize Jerusalem as Israel’s capital means for the city, the conflict and the world https://t.co/TMzXarFvfL
    {'neg': 0.113, 'neu': 0.887, 'pos': 0.0, 'compound': -0.3182}
    Sun Dec 10 11:00:12 +0000 2017
     
    Janet Yellen Didn’t Set Out to Be a Feminist Hero https://t.co/ZBBXFKq6qG
    {'neg': 0.0, 'neu': 0.714, 'pos': 0.286, 'compound': 0.5574}
    Sun Dec 10 10:58:30 +0000 2017
     
    How Luke Bryan and Blake Shelton, 2 of Nashville’s biggest stars, are adapting to country’s new ways https://t.co/v9ZM6v9Rqr
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 10:36:14 +0000 2017
     
    She survived a brain tumor at age 4. It was just the start of her fight. https://t.co/JmOzuWU2wz
    {'neg': 0.242, 'neu': 0.605, 'pos': 0.153, 'compound': -0.2263}
    Sun Dec 10 10:19:16 +0000 2017
     
    A new way of measuring school performance turns up some unexpected winners https://t.co/gU9YlIGWja
    {'neg': 0.0, 'neu': 0.78, 'pos': 0.22, 'compound': 0.4767}
    Sun Dec 10 10:02:12 +0000 2017
     
    The Power of Touch, Especially for Men https://t.co/ZM0Vsniy2Z
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 09:43:19 +0000 2017
     
    After the death of a young son, can a long road trip help a family to heal? https://t.co/PqFvd8sU6r
    {'neg': 0.199, 'neu': 0.663, 'pos': 0.138, 'compound': -0.296}
    Sun Dec 10 09:29:06 +0000 2017
     
    Tens of thousands of people said goodbye and "Merci, Johnny" to the French rocker Johnny Hallyday in Paris https://t.co/NIAg0tdWJa
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 09:10:03 +0000 2017
     
    ‘S.N.L.,’ Santa and James Franco Relentlessly Tackle Sexual Harassment https://t.co/kgdLeYncPO
    {'neg': 0.28, 'neu': 0.72, 'pos': 0.0, 'compound': -0.5423}
    Sun Dec 10 08:53:33 +0000 2017
     
    "To me, the college send-off was not a blues for Chloe. It was a celebration." https://t.co/btkS6LZCJt
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 08:50:24 +0000 2017
     
    He defected from North Korea more than 3 decades ago. So why won't the South accept him? https://t.co/ojIaNT0KB1
    {'neg': 0.246, 'neu': 0.754, 'pos': 0.0, 'compound': -0.5972}
    Sun Dec 10 08:34:21 +0000 2017
     
    Ruby Rose has a whole closet dedicated to skin and hair care. "I consider this all self-care," she said. https://t.co/mGUBF2B6h9
    {'neg': 0.0, 'neu': 0.733, 'pos': 0.267, 'compound': 0.7351}
    Sun Dec 10 08:15:10 +0000 2017
     
    Actually, you do want to know how this sausage gets made https://t.co/SDmBZBmDkJ via @NYTScience
    {'neg': 0.0, 'neu': 0.909, 'pos': 0.091, 'compound': 0.0772}
    Sun Dec 10 07:59:01 +0000 2017
     
    8 ways to have a better relationship in 2018 https://t.co/VlRuKBEKvC
    {'neg': 0.0, 'neu': 0.707, 'pos': 0.293, 'compound': 0.4404}
    Sun Dec 10 07:40:01 +0000 2017
     
    An algorithm analyzed 52 million dolphin clicks. It found 7 distinct types of sounds. https://t.co/noyluyBGrH
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 07:20:51 +0000 2017
     
    Older Venezuelans who have migrated say that arguably the biggest hardship is starting over in their sunset years https://t.co/Z9RfqIqtuH
    {'neg': 0.202, 'neu': 0.798, 'pos': 0.0, 'compound': -0.5106}
    Sun Dec 10 07:04:24 +0000 2017
     
    How to use an Instant Pot https://t.co/1nJIGi8N8Y https://t.co/C2Ic2vzmmg
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 06:48:22 +0000 2017
     
    Macron Steps Into Middle East Role as U.S. Retreats https://t.co/mSiyzgup8j
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 06:29:54 +0000 2017
     
    These almond cookies need just 4 ingredients https://t.co/WxR0qJfsHF
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 05:56:32 +0000 2017
     
    It has been several generations since Haiti was a major tourist destination, but it may become one again https://t.co/JpUVdMThi8
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 05:37:35 +0000 2017
     
    Many athletes have been told that smiling while sweating will make our efforts feel easier. But is it actually true? https://t.co/i9wPDL8lm8
    {'neg': 0.0, 'neu': 0.677, 'pos': 0.323, 'compound': 0.8225}
    Sun Dec 10 05:21:08 +0000 2017
     
    The best TV shows of 2017 https://t.co/b6UzsqJTb8
    {'neg': 0.0, 'neu': 0.588, 'pos': 0.412, 'compound': 0.6369}
    Sun Dec 10 05:04:51 +0000 2017
     
    Trump reduced the size of 2 monuments this week and plans for many others are unclear. We explored a few of those p… https://t.co/lXFtPoN4kO
    {'neg': 0.087, 'neu': 0.913, 'pos': 0.0, 'compound': -0.25}
    Sun Dec 10 04:47:15 +0000 2017
     
    Here's how to get your body and mind in top shape in 2018, whatever that means for you https://t.co/xcs0zL1z12
    {'neg': 0.0, 'neu': 0.909, 'pos': 0.091, 'compound': 0.2023}
    Sun Dec 10 04:30:14 +0000 2017
     
    Maybe your love of “Star Wars” runs deep. Maybe you're a newbie. We catch you up ahead of "The Last Jedi."… https://t.co/tJHSnELSGV
    {'neg': 0.0, 'neu': 0.826, 'pos': 0.174, 'compound': 0.6369}
    Sun Dec 10 04:11:22 +0000 2017
     
    26 casseroles for cold nights https://t.co/K2t598jCUP https://t.co/q5OEUXrScn
    {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    Sun Dec 10 03:53:54 +0000 2017
     
    In “Home,” an entire house is custom built, room by room, on stage before your astonished eyes https://t.co/XpinY2NTHL
    {'neg': 0.0, 'neu': 0.867, 'pos': 0.133, 'compound': 0.3818}
    Sun Dec 10 03:35:12 +0000 2017
     
    Here’s how to become part of the small group of people that successfully achieve their New Year's resolution https://t.co/oBxWcJxlTM
    {'neg': 0.0, 'neu': 0.849, 'pos': 0.151, 'compound': 0.4939}
    Sun Dec 10 03:18:33 +0000 2017
     



```python
nyt_complete_df = pd.DataFrame({'Date (New York Times)': nyt_date_list, 'Tweet (New York Times)': nyt_text_list, 
                                'Negative Score (New York Times)': nyt_negative_list, 
                                'Neutral Score (New York Times)': nyt_neutral_list, 
                                'Positive Score (New York Times)': nyt_positive_list, 
                                'Compound Score (New York Times)': nyt_compound})
nyt_final_df = nyt_complete_df[['Date (New York Times)', 'Tweet (New York Times)', 
                                'Negative Score (New York Times)', 'Neutral Score (New York Times)', 
                                'Positive Score (New York Times)', 'Compound Score (New York Times)']]
nyt_final_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date (New York Times)</th>
      <th>Tweet (New York Times)</th>
      <th>Negative Score (New York Times)</th>
      <th>Neutral Score (New York Times)</th>
      <th>Positive Score (New York Times)</th>
      <th>Compound Score (New York Times)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mon Dec 11 04:12:09 +0000 2017</td>
      <td>Your cat tattoo can have your actual cat in it...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mon Dec 11 03:53:03 +0000 2017</td>
      <td>RT @nytimesworld: Protests in Lebanon near U.S...</td>
      <td>0.128</td>
      <td>0.872</td>
      <td>0.000</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon Dec 11 03:39:09 +0000 2017</td>
      <td>Horror films dominated the cultural conversati...</td>
      <td>0.143</td>
      <td>0.695</td>
      <td>0.162</td>
      <td>0.1280</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mon Dec 11 03:25:04 +0000 2017</td>
      <td>RT @nytimesworld: London's Heathrow Airport wa...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mon Dec 11 03:13:26 +0000 2017</td>
      <td>17 ways to celebrate the season in New York Ci...</td>
      <td>0.000</td>
      <td>0.816</td>
      <td>0.184</td>
      <td>0.6467</td>
    </tr>
  </tbody>
</table>
</div>




```python
nyt_compound_df = pd.DataFrame(nyt_compound)
nyt_compound_df.columns = ['New York Times Compound Score']
nyt_compound_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>New York Times Compound Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1280</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6467</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweet_number = np.arange(1,101)
combined_compound_df = pd.DataFrame({'# Tweet(s) Ago': tweet_number, 'BBC Compound Score': bbc_compound,
                      'CBS Compound Score': cbs_compound,'CNN Compound Score': cnn_compound,
                      'Fox Compound Score': fox_compound,'New York Times Compound Score': nyt_compound})
combined_compound_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Tweet(s) Ago</th>
      <th>BBC Compound Score</th>
      <th>CBS Compound Score</th>
      <th>CNN Compound Score</th>
      <th>Fox Compound Score</th>
      <th>New York Times Compound Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.6915</td>
      <td>-0.2263</td>
      <td>0.2023</td>
      <td>-0.4019</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.0000</td>
      <td>-0.2263</td>
      <td>0.0000</td>
      <td>0.6369</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.4391</td>
      <td>0.0772</td>
      <td>0.4019</td>
      <td>0.5267</td>
      <td>0.1280</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-0.3818</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.3818</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.0000</td>
      <td>0.5754</td>
      <td>0.0000</td>
      <td>-0.4588</td>
      <td>0.6467</td>
    </tr>
  </tbody>
</table>
</div>




```python
reversed_df = combined_compound_df.sort_index(ascending=False)
reversed_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Tweet(s) Ago</th>
      <th>BBC Compound Score</th>
      <th>CBS Compound Score</th>
      <th>CNN Compound Score</th>
      <th>Fox Compound Score</th>
      <th>New York Times Compound Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>100</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.4026</td>
      <td>0.0000</td>
      <td>0.4939</td>
    </tr>
    <tr>
      <th>98</th>
      <td>99</td>
      <td>0.4939</td>
      <td>0.6289</td>
      <td>0.8360</td>
      <td>0.2732</td>
      <td>0.3818</td>
    </tr>
    <tr>
      <th>97</th>
      <td>98</td>
      <td>0.6808</td>
      <td>0.9476</td>
      <td>0.0000</td>
      <td>0.7003</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>96</th>
      <td>97</td>
      <td>0.0000</td>
      <td>0.5106</td>
      <td>0.0000</td>
      <td>0.4939</td>
      <td>0.6369</td>
    </tr>
    <tr>
      <th>95</th>
      <td>96</td>
      <td>0.0000</td>
      <td>0.5826</td>
      <td>0.0000</td>
      <td>-0.8074</td>
      <td>0.2023</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter(reversed_df['# Tweet(s) Ago'], reversed_df['BBC Compound Score'], color='c')
plt.scatter(reversed_df['# Tweet(s) Ago'], reversed_df['CBS Compound Score'], color='g')
plt.scatter(reversed_df['# Tweet(s) Ago'], reversed_df['CNN Compound Score'], color='r')
plt.scatter(reversed_df['# Tweet(s) Ago'], reversed_df['Fox Compound Score'], color='b')
plt.scatter(reversed_df['# Tweet(s) Ago'], reversed_df['New York Times Compound Score'], color='y')
plt.xlim([105,-5])
plt.ylim([-1.0, 1.0])
plt.xlabel('Tweets Ago')
plt.ylabel('Tweet Polarity')
plt.title('Sentiment Analysis of Media Tweets (12/10/17)')

bbc_legend = mpatches.Patch(color='c', label='BBC')
cbs_legend = mpatches.Patch(color='g', label='CBS')
cnn_legend = mpatches.Patch(color='r', label='CNN')
fox_legend = mpatches.Patch(color='b', label='Fox')
nyt_legend = mpatches.Patch(color='y', label='New York Times')

plt.legend(handles=[bbc_legend, cbs_legend, cnn_legend, fox_legend, nyt_legend], 
           loc='upper left', bbox_to_anchor=(1,1), title='Media Sources')

sns.set_style("whitegrid")
plt.savefig('Sentiment_Analysis_of_Media_Tweets.png')
plt.show()
```


![png](output_21_0.png)



```python
means_df = pd.DataFrame({'BBC Compound Score': bbc_compound,
                      'CBS Compound Score': cbs_compound,'CNN Compound Score': cnn_compound,
                      'Fox Compound Score': fox_compound,'New York Times Compound Score': nyt_compound})

means = means_df.mean()
means
```




    BBC Compound Score               0.170004
    CBS Compound Score               0.374197
    CNN Compound Score              -0.009598
    Fox Compound Score               0.059802
    New York Times Compound Score    0.022462
    dtype: float64




```python
accounts = ['BBC','CBS','CNN', 'Fox', 'NYT']
data = [means['BBC Compound Score'], means['CBS Compound Score'], means['CNN Compound Score'],
        means['Fox Compound Score'], means['New York Times Compound Score']]
labels = [str(means['BBC Compound Score']),str(means['CBS Compound Score']), str(means['CNN Compound Score']), 
          str(means['Fox Compound Score']),str(means['New York Times Compound Score'])]
colors = ['c','g','r','b','y']

plt.bar(accounts, data, color=colors, width = 1)
plt.ylim([-0.20,0.50])
plt.ylabel('Tweet Polarity')
plt.title('Overall Media Sentiment based on Twitter (12/10/17)')


bbc_legend = mpatches.Patch(color='c', label= means['BBC Compound Score'])
cbs_legend = mpatches.Patch(color='g', label= means['CBS Compound Score'])
cnn_legend = mpatches.Patch(color='r', label= means['CNN Compound Score'])
fox_legend = mpatches.Patch(color='b', label= means['Fox Compound Score'])
nyt_legend = mpatches.Patch(color='y', label= means['New York Times Compound Score'])

plt.legend(handles=[bbc_legend, cbs_legend, cnn_legend, fox_legend, nyt_legend], 
           loc='upper left', bbox_to_anchor=(1,1), title='Values')
sns.set_style("whitegrid")
plt.savefig('Overall_Media_Sentiment.png')
plt.show()
```


![png](output_23_0.png)



```python
everything_df = pd.DataFrame({'Date (BBC)': bbc_date_list, 'Tweet (BBC)': bbc_text_list, 
                              'Negative Score (BBC)': bbc_negative_list, 'Neutral Score (BBC)': bbc_neutral_list, 
                              'Positive Score (BBC)': bbc_positive_list, 'Compound Score (BBC)': bbc_compound,
                              'Date (CBS)': cbs_date_list, 'Tweet (CBS)': cbs_text_list, 
                              'Negative Score (CBS)': cbs_negative_list, 'Neutral Score (CBS)': cbs_neutral_list, 
                              'Positive Score (CBS)': cbs_positive_list, 'Compound Score (CBS)': cbs_compound,
                              'Date (CNN)': cnn_date_list, 'Tweet (CNN)': cnn_text_list, 
                              'Negative Score (CNN)': cnn_negative_list, 'Neutral Score (CNN)': cnn_neutral_list, 
                              'Positive Score (CNN)': cnn_positive_list, 'Compound Score (CNN)': cnn_compound,
                              'Date (Fox)': fox_date_list, 'Tweet (Fox)': fox_text_list, 
                              'Negative Score (Fox)': fox_negative_list, 'Neutral Score (Fox)': fox_neutral_list, 
                              'Positive Score (Fox)': fox_positive_list, 'Compound Score (Fox)': fox_compound,
                              'Date (New York Times)': nyt_date_list, 'Tweet (New York Times)': nyt_text_list, 
                              'Negative Score (New York Times)': nyt_negative_list, 
                              'Neutral Score (New York Times)': nyt_neutral_list, 
                              'Positive Score (New York Times)': nyt_positive_list, 
                              'Compound Score (New York Times)': nyt_compound})
last_df = everything_df[['Date (BBC)', 'Tweet (BBC)', 'Negative Score (BBC)', 'Neutral Score (BBC)', 
                          'Positive Score (BBC)', 'Compound Score (BBC)','Date (CBS)', 'Tweet (CBS)', 
                          'Negative Score (CBS)', 'Neutral Score (CBS)', 'Positive Score (CBS)', 
                          'Compound Score (CBS)', 'Date (CNN)', 'Tweet (CNN)', 'Negative Score (CNN)', 
                          'Neutral Score (CNN)', 'Positive Score (CNN)', 'Compound Score (CNN)', 'Date (Fox)', 
                          'Tweet (Fox)', 'Negative Score (Fox)', 'Neutral Score (Fox)', 'Positive Score (Fox)', 
                          'Compound Score (Fox)', 'Date (New York Times)', 'Tweet (New York Times)', 
                          'Negative Score (New York Times)', 'Neutral Score (New York Times)', 
                          'Positive Score (New York Times)', 'Compound Score (New York Times)']]
last_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date (BBC)</th>
      <th>Tweet (BBC)</th>
      <th>Negative Score (BBC)</th>
      <th>Neutral Score (BBC)</th>
      <th>Positive Score (BBC)</th>
      <th>Compound Score (BBC)</th>
      <th>Date (CBS)</th>
      <th>Tweet (CBS)</th>
      <th>Negative Score (CBS)</th>
      <th>Neutral Score (CBS)</th>
      <th>...</th>
      <th>Negative Score (Fox)</th>
      <th>Neutral Score (Fox)</th>
      <th>Positive Score (Fox)</th>
      <th>Compound Score (Fox)</th>
      <th>Date (New York Times)</th>
      <th>Tweet (New York Times)</th>
      <th>Negative Score (New York Times)</th>
      <th>Neutral Score (New York Times)</th>
      <th>Positive Score (New York Times)</th>
      <th>Compound Score (New York Times)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sun Dec 10 21:34:02 +0000 2017</td>
      <td>RT @BBCOne: SO. MUCH. CUTE. 😍\n#Attenboroughan...</td>
      <td>0.000</td>
      <td>0.560</td>
      <td>0.440</td>
      <td>0.6915</td>
      <td>Mon Dec 11 00:26:50 +0000 2017</td>
      <td>Due to NFL overrun, CBS is delayed 8 mins in t...</td>
      <td>0.083</td>
      <td>0.917</td>
      <td>...</td>
      <td>0.144</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>-0.4019</td>
      <td>Mon Dec 11 04:12:09 +0000 2017</td>
      <td>Your cat tattoo can have your actual cat in it...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun Dec 10 21:02:27 +0000 2017</td>
      <td>RT @BBCEarth: 'Never before have we had such a...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>Mon Dec 11 00:25:54 +0000 2017</td>
      <td>Due to NFL overrun CBS is delayed 7 mins in th...</td>
      <td>0.087</td>
      <td>0.913</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.826</td>
      <td>0.174</td>
      <td>0.6369</td>
      <td>Mon Dec 11 03:53:03 +0000 2017</td>
      <td>RT @nytimesworld: Protests in Lebanon near U.S...</td>
      <td>0.128</td>
      <td>0.872</td>
      <td>0.000</td>
      <td>-0.2263</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sun Dec 10 21:00:06 +0000 2017</td>
      <td>🌹@DuaLipa performing 'Homesick' was a complete...</td>
      <td>0.000</td>
      <td>0.855</td>
      <td>0.145</td>
      <td>0.4391</td>
      <td>Sun Dec 10 22:49:02 +0000 2017</td>
      <td>RT @NoActivityCBS: If you want the intel, you ...</td>
      <td>0.000</td>
      <td>0.936</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.825</td>
      <td>0.175</td>
      <td>0.5267</td>
      <td>Mon Dec 11 03:39:09 +0000 2017</td>
      <td>Horror films dominated the cultural conversati...</td>
      <td>0.143</td>
      <td>0.695</td>
      <td>0.162</td>
      <td>0.1280</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sun Dec 10 20:58:15 +0000 2017</td>
      <td>RT @BBCEarth: 'What shocks me ...is how fast t...</td>
      <td>0.126</td>
      <td>0.874</td>
      <td>0.000</td>
      <td>-0.3818</td>
      <td>Sun Dec 10 22:48:46 +0000 2017</td>
      <td>RT @startrekcbs: .@albinokid and @wcruz73 are ...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.860</td>
      <td>0.140</td>
      <td>0.3818</td>
      <td>Mon Dec 11 03:25:04 +0000 2017</td>
      <td>RT @nytimesworld: London's Heathrow Airport wa...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sun Dec 10 20:57:07 +0000 2017</td>
      <td>RT @BBCOne: If we don’t act, coral reefs could...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>Sat Dec 09 18:24:37 +0000 2017</td>
      <td>Don’t miss America’s Game! Stream the Army-Nav...</td>
      <td>0.070</td>
      <td>0.742</td>
      <td>...</td>
      <td>0.176</td>
      <td>0.824</td>
      <td>0.000</td>
      <td>-0.4588</td>
      <td>Mon Dec 11 03:13:26 +0000 2017</td>
      <td>17 ways to celebrate the season in New York Ci...</td>
      <td>0.000</td>
      <td>0.816</td>
      <td>0.184</td>
      <td>0.6467</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
last_df.to_csv('Media_Sentiment_Scores.csv', encoding='utf-8')
```
