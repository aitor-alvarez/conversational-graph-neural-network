import tweepy
import pandas as pd
import creds
import time

tkn = getattr(creds, "Bearer_Token", None)


#Download tweets using the Twitter research API (hence max 300 tweets in 15 minutes) from a list of ids
def download_tweets(data_file, output_file):
	df = pd.read_csv(data_file)
	tweets = []
	client = tweepy.Client(bearer_token=tkn)
	start = 0
	for row in df.iterrows():
		start+=1
		if start<300:
			try:
				twt = client.get_tweet(row[1][0])
				txt = twt.data.text
			except:
				txt = None
			tweets.append(txt)
		else:
			time.sleep(900)
			start=0
	df['tweet_text'] = tweets
	df.to_csv(output_file)