# English-and-Thai-OCR-Machine-Learning-Task

The order of running the scripts is : 
1. data_split_script.py
    There are 4 mandatory arguments and 3 optional in the following order : 
        Train language which is a string and can be English, Thai or both. 
        Train font which is a string and can be normal, bold, bold_italic, italic, or all. 
        Train dpi the resolution of the image which is an integer and can be 200,300, or 400 or all. 
        The last mandatory argument is the directory where the data is which I provide in the help information as well.
        The optional arguments are useful for when the train and testing data are different and the options for those are the same. 
        In order to use the optional arguments they need to be called
        explicitly as --test_language --test_font --test_dpi and the argument of each which are the same.
    e.g. Same data for training and testing : python3 data_split_script.py English normal 200 /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet
    e.g. Different data for training and testing : python3 data_split_script.py English normal 200 /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet --test_language English --test_font bold --test_dpi 200 

2. train.py
    There are 2 mandatory arguments in the following order :
        batch_size which can be an integer but recommended is to use an even integer.
        num_epochs which can be an integer but recommended is above 10.

3. validation.py (optional)
    There is only one mandatory arguments :
        batch_size which can be an integer it should be the same as training.

4. test.py
    There is one mandatory argument :
        batch_size which can be an integer it should be the same as training.
    There is also an optional argument:
        show errors that was added to do the qualitative error analysis.

Challenges and Design decisions:
    The biggest challenge was making the splitting data script as it had to incorporate a lot of different choices and there were
    issues like dealing with bold and bold_italic which were often wrongfully chosen together and having different train and test data options.
    The second biggest challenge was dealing with the images. Data manipulation and finding a good padding, resizing ratio was a challenge.
    Initially, I wanted to dynamically choose the target size depending on the max width and height of each resolution and font style
    but that would mean having more parameters that would also affect the model layers because of max pooling for example (some numbers can't be
    divided by 2 twice for example). So I decided to simplify my logic and choose a middle ground target size of 60,60 
    while adding white padding around the image and resizing in the cases where the images are bigger. I would like to try to change this
    in the future but as for now I left it as is. The model has 2 convolutional layers each coming with a non-linear layer and max pooling.
    There is also dropout and batch normalization to prevent overfitting.

Experiments : 
 I decided to run the experiments with the same batch size and epochs so I can compare how the model performs without necessarily 
 tuning the hyperparameters to get the best results.
    1. Train : English normal 200 , Test : English normal 200
        Batch size : 8 , epochs : 15
        The first experiment that I run was English. The model performs really well above 90%.
        Accuracy, precision and recall is 92% and f1 score is 91%.

    2. Train : Thai normal 200 , Test : Thai normal 200
        Batch size : 8 , epochs : 15
        The model performs exceptionally good the results on the testing data are accuracy 94% , precision 95 %, recall 94% and F1 score 94%. So the model has learned how to recognize the characters. I think the model performs a bit better than English because there is more data available.
    
    3. Train : Thai normal 400 , Test : Thai normal 200
        Batch size : 8 , epochs : 15
        The model performs considerably less better. The results are accuracy and recall 70%, precision 78% and f1 score 71%. The model is still doing good but the difference in training and testing data affected the results. I ran the same experiment again with 20 epochs but the results were quite similar only 1 percent higher. I believe that in this case if there was dynamic padding the model would perform better as it would find a more appropriate target size.
    
    4. Train : Thai normal 400, Test : Thai bold 400
        Batch size : 8 , epochs : 15
        The model performs as good as before as the results are accuracy, recall, precision and f1 score 93%. The model generalizes really well and has no problem with the change from normal to bold which also makes sense as the shape is the same just a little bit more black if I describe it in a simplified way.

    5. Train : Thai bold 400 , Test : Thai normal 400
        Batch size : 8 , epochs : 15
        I chose the resolution 400 to check in comparison with the previous experiment how the model performs. The model generalizes
        succesfully as well as it performs the same as in experiment 4. The results are accuracy,recall and f1 score 93% and presicion 94%. I think max pooling is the reason that the model performs so well in both cases. I also run the experiment with all resolutions to check if the model performs better since there are more data or if it gets confused by the change of resolution like it happened in experiment 3. The results were very similar and there was only a 1% increase and I noticed from the model that it stopped learning in around the 10th epoch. I think the model because of the depth of the layers learned pretty fast which is beneficial when it comes to computational cost but perhaps has the limitation of being hard to perfect if it's even possible to get higher results.

    6. Train : Thai all styles 200 , Test : Thai all styles 200
        Batch size : 8 , epochs : 15
        The results of this experiment were very disappointing and proved that the model architecture isn't deep enough to deal with more complicated tasks of OCR. All of the evaluation metrics were below 1%. I checked to make sure there is no fault in the training or in the way of collecting data and it seemed like the model was never learning as the loss was not decreasing so even if there were more epochs the result would probably be the same. In this case it's important to reconsider the model architecture. However, I didn't change it. Surprisingly when running the test with 400 dpi instead of 200, the model actually excels and has 97% on the evaluation. My theory is that the padding of target 60 is too high for the 200 dpi so it can't focus and learn the information.
    
    7. Train : Both languages normal all , Test : Both languages normal all
        Batch size : 8 , epochs : 15
        It seems like even though in this experiment there are a lot of data and a lot of labels (151) because of the size of the data the model generalizes really good. The results of the testing was 96% for all evaluation metrics which is quite high. Maybe it performed this well because the style was consistent.
    
    8. Train : Both languages all styles all, Test : Both languages all styles all
        Batch size : 8 , epochs : 15
        I thought that the model wouldn't perform as well in this case because of all the different styles and resolutions but it actually succeeded with results of 95%. This means that the model is really good at generalizing even with 15 epochs as long as there is enough data. 
    
    9. Train :  English all all , Test : English all styles all
        Batch size : 8 , epochs : 15
        The model seems to be performing consistently with a 93% on evaluations metrics which proves once again that the model architecture in conjuction with the image manipulations is adequate enough to have really good results. I used the optional show errors argument in the testing data to analyze and figure out what the model isn't learning. (For me the output isn't shown when run on the terminal not sure why I think it's a windows issue so I opened a jupyter notebook and run it there to see the outputs.)
        My observations are that the model struggles in recognizing the following labels 120 which is the letter "x" , 073 which is "i", 119 which is "w", 111 which is "o", 089 which is "y", 118 which is "v" and 083 which is the letter "s". When scrolling through the resuts those were the majority of the errors however there were times that were recognised correctly. The issue seemed to arise mostly in italics and bold. Italics wasn't a surprise as when inspecting the images they looked a bit deformed with the padding and sometimes the letter itself was quite small when compared to the ratio of padding. I can't really figure out what went wrong with bold but perhaps an idea is that some of the bold dpi 400 images were a lot bigger than 60 x 60 so when resized to fit the dimensions it was confusing  the model.
    

