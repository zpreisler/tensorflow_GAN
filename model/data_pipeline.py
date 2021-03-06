def _parse_fce(file):
    from numpy.random import uniform
    import tensorflow as tf
    img_str=tf.read_file(file)
    img_decoded=tf.image.decode_png(img_str,channels=1)
    #img_crop=tf.image.central_crop(img_decoded,0.5)
    #img_resized=tf.image.resize_images(img_crop,[64,64])
    #img_resized=tf.image.resize_images(img_crop,[64,64])
    #return img_resized/255.0
    img_resized=tf.image.resize_images(img_decoded,[64,64])
    return img_resized

def dataset(files_str='/home/zdenek/Projects/tensorflow/patchy_ann/data_4/*.png',batch_size=1):
    from glob import glob
    import tensorflow as tf

    files=glob(files_str)
    files_dataset=tf.data.Dataset.from_tensor_slices((files))
    files_dataset=files_dataset.map(_parse_fce)

    dataset=files_dataset.repeat().batch(batch_size)
    #dataset=files_dataset.batch(batch_size)
    iterator=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)

    img_batch=iterator.get_next()
    init_dataset=iterator.make_initializer(dataset)

    return img_batch,init_dataset
