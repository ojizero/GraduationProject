def pipeline (data, *transformers):
	for transformer in transformers:
		data = transformer(data)
	return data