# Gen training data with all charactor
python3 image_gen.py --fontlist Japanese_fonts_train.txt --labellist all_charactor.txt --output ../dataset/training_dataset --worker 4
# Gen test data with all charactor
python3 image_gen.py --fontlist Japanese_fonts_test.txt --labellist all_charactor.txt --output ../dataset/testing_dataset --worker 4

# Gen training data with digit only
python3 image_gen.py --fontlist Japanese_fonts_train.txt --labellist digit.txt --output ../dataset/training_dataset --worker 4
# Gen test data with digit only
python3 image_gen.py --fontlist Japanese_fonts_test.txt --labellist digit.txt --output ../dataset/testing_dataset --worker 4

# Gen training data with hiragana only
python3 image_gen.py --fontlist Japanese_fonts_train.txt --labellist hiragana.txt --output ../dataset/training_dataset --worker 4
# Gen test data with hiragana only
python3 image_gen.py --fontlist Japanese_fonts_test.txt --labellist hiragana.txt --output ../dataset/testing_dataset --worker 4

# Gen training data with katakana only
python3 image_gen.py --fontlist Japanese_fonts_train.txt --labellist katakana.txt --output ../dataset/training_dataset --worker 4
# Gen test data with katakana only
python3 image_gen.py --fontlist Japanese_fonts_test.txt --labellist katakana.txt --output ../dataset/testing_dataset --worker 4

# Gen training data with kanji only
python3 image_gen.py --fontlist Japanese_fonts_train.txt --labellist kanji.txt --output ../dataset/training_dataset --worker 4
# Gen test data with kanji only
python3 image_gen.py --fontlist Japanese_fonts_test.txt --labellist kanji.txt --output ../dataset/testing_dataset --worker 4
