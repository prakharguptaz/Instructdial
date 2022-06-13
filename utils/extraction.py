import yake
kw_extractor = yake.KeywordExtractor()

#pip install git+https://github.com/LIAAD/yake
def get_keywords(text, numOfKeywords=20):
	global kw_extractor
	if numOfKeywords!=20:
		kw_extractor = yake.KeywordExtractor(top=numOfKeywords)
	keywords = kw_extractor.extract_keywords(text)

	return keywords
