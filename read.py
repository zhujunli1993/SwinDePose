import tensorflow as tf
for example in tf.python_io.tf_record_iterator("0_syn.tfrecords"):
    result = tf.train.SequenceExample.FromString(example)
    
    print(result)
