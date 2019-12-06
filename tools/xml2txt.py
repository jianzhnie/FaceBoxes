import xml.etree.ElementTree as ET
import os

classes = ["0", "face", "body"]
root_dir = "/home/expert/Downloads/NVR_v1.2"

imgs_dir = os.path.join(root_dir,"NVR_v1.2_images")
anno_dir = os.path.join(root_dir,"NVR_v1.2_anno")

out_file = open(os.path.join(root_dir,"NVR_v1.2_val_fddb.txt"), 'w')

def convert_annotation(image_id):
    if True:
        in_file = open(os.path.join(anno_dir,'%s.xml' % image_id))
        img_name = os.path.join(imgs_dir,"{}.jpg".format(image_id))
        out_file.write("%s"%(img_name))
        tree = ET.parse(in_file)
        root = tree.getroot()
        box = []

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            #obj = obj.find('bndbox').text
            obj = obj.find('bndbox')

            #bb = [float(obj.find('xmin').text), float(obj.find('ymin').text), float(obj.find('xmax').text) - float(obj.find('xmin').text), float(obj.find('ymax').text) - float(obj.find('ymin').text), cls_id]
            bb = [float(obj.find('xmin').text), float(obj.find('ymin').text), float(obj.find('xmax').text), float(obj.find('ymax').text), cls_id]
            box.append(bb)

        out_file.write(" %d"%len(box))
        for bb in box:
            out_file.write(" %d %d %d %d %d"%(bb[0], bb[1], bb[2], bb[3], bb[4]))
        out_file.write("\n")

#filelist = os.listdir(imgs_dir)
imf = open(os.path.join(root_dir, "NVR_v1.2_val.txt"), 'r')
filelist = [mk.strip() for mk in imf.readlines()]

for files in filelist:
    filename0 = os.path.splitext(files)[0] 
    convert_annotation(filename0)

