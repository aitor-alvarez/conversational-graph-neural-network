import json
import os
import pandas as pd

dataset = 'data/pheme-rnr-dataset/'

def process_pheme(dataset):
	for event in os.listdir(dataset):
		data = pd.DataFrame()
		if os.path.isdir(dataset+event):
			fn = "data/pheme-rnr-dataset/%s.csv" % (event)
			for category in os.listdir("%s/%s" % (dataset, event)):
				if os.path.isdir(dataset + event+'/'+category+'/'):
					for thread in os.listdir("%s/%s/%s" % (dataset, event, category)):
						if '.DS_Store' not in thread:
							with open("%s/%s/%s/%s/source-tweet/%s.json" % (dataset, event, category, thread, thread)) as f:
								tweet = json.load(f)
							df = tweet_to_df(tweet, category, thread)
							data = data.append(df)
							for reaction in os.listdir("%s/%s/%s/%s/reactions" % (dataset, event, category, thread)):
								if '.DS_Store' not in reaction:
									with open("%s/%s/%s/%s/reactions/%s" % (dataset, event, category, thread, reaction)) as f:
										tweet = json.load(f)
									df = tweet_to_df(tweet, category, thread, False)
									data = data.append(df)
			data.to_csv(fn, index=False)
	return None


def tweet_to_df(twt, cat, thrd, is_source_tweet=True):
	return pd.DataFrame([{
		"thread": thrd,
		"tweet_length": len(twt.get("text", "")),
		"text": twt.get("text"),
		"id": twt.get("id"),
		"in_reply_id": twt.get("in_reply_to_status_id", None),
		"in_reply_user": twt.get("in_reply_to_user", None),
		"is_rumor": True if cat == "rumours" else False,
		"is_source_tweet": is_source_tweet,
		"created": twt.get("created_at"),
	}])