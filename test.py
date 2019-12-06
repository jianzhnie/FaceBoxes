from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data.config import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
from PIL import Image
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
import time
import json, codecs


headstr = """\
<annotation>
    <folder>VOC2007</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
    </source>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
        <score>%f</score>
    </object>
"""

tailstr = '''\
</annotation>
'''


parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--model', default='weights/weights_fbFinal_FaceBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--outformat', default='WIDER', type=str, choices=['WIDER', 'VOC', 'FDDB', 'JSON'], help='outformat')
parser.add_argument('--confidence_threshold', default=0.09, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--test_list', default='None', type=str, help='images path list')
parser.add_argument('--test_dir', default='None', type=str, help='images path dir')
parser.add_argument('--half', default=True, help='test on half mode or fp32')
parser.add_argument('--face_thresh', default=0.1, type=float, help='face_threshold')
parser.add_argument('--body_thresh', default=0.1, type=float, help='body_threshold')
args = parser.parse_args()

conf_thresh = {'face':args.face_thresh, 'body':args.body_thresh}

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# save file
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

if args.outformat == "FDDB":
    fwf = open(os.path.join(args.save_folder, os.path.splitext(os.path.basename(args.model))[0] + '_face_dets.txt'), 'w')
    fwb = open(os.path.join(args.save_folder, os.path.splitext(os.path.basename(args.model))[0] + '_body_dets.txt'), 'w')
if args.outformat == 'WIDER':
    fw = open(os.path.join(args.save_folder, os.path.splitext(os.path.basename(args.model))[0] + '_dets.txt'), 'w')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=3)    # initialize detector
    net = load_model(net, args.model, args.cpu)
    net.eval()
    print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    if args.half:
        net = net.to(device).half()
    else:
        net = net.to(device)

    # testing dataset
    if args.test_list is not "None":
        fr = open(args.test_list, 'r')
        test_dataset = [mk.strip() for mk in fr.readlines()]

    if args.test_dir is not "None":    
        imgs = os.listdir(args.test_dir)
        test_dataset = [os.path.join(args.test_dir, mk) for mk in imgs]

    num_images = len(test_dataset)

    # testing scale
    resize = 1
    _t = {'forward_pass': Timer(), 'misc': Timer()}

    face_, body_ = {}, {}

    # testing begin
    for i, img_name in enumerate(test_dataset):
        tt_0 = time.time()
        #image_path = testset_folder + img_name + '.jpg'
        image_path = img_name
        #img = np.float32(cv2.imread(image_path, cv2.IMREAD_COLOR))
        img = np.float32(Image.open(image_path))
        #if resize != 1:
        #    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (123, 117, 104)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        if args.half:
            img = img.to(device).half()
            scale = scale.to(device).half()
            priors = priors.to(device).half()
        else:
            img = img.to(device)
            scale = scale.to(device)
            priors = priors.to(device)

        prior_data = priors.data

        tt_00 = time.time()
        print("image process:", tt_00-tt_0)
        _t['forward_pass'].tic()
        t1 =time.time()
        #torch.cuda.synchronize()
        loc, conf = net(img)  # forward pass
        torch.cuda.synchronize()
        t2 =time.time()
        print("inference",t2-t1)
        #_t['forward_pass'].toc()
        #_t['misc'].tic()
        torch.cuda.synchronize()
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        face_scores = conf.data.cpu().numpy()[:, 1]
        body_scores = conf.data.cpu().numpy()[:, 2]

        # ignore low scores
        #face_inds = np.where(face_scores > args.confidence_threshold)[0]
        #body_inds = np.where(body_scores > args.confidence_threshold)[0]
        face_inds = np.where(face_scores > conf_thresh['face'])[0]
        body_inds = np.where(body_scores > conf_thresh['body'])[0]
        face_boxes = boxes[face_inds]
        body_boxes = boxes[body_inds]

        face_scores = face_scores[face_inds]
        body_scores = body_scores[body_inds]

        # keep top-K before NMS
        order = face_scores.argsort()[::-1][:args.top_k]
        face_boxes = face_boxes[order]
        face_scores = face_scores[order]

        order = body_scores.argsort()[::-1][:args.top_k]
        body_boxes = body_boxes[order]
        body_scores = body_scores[order]


        # do NMS
        face_dets = np.hstack((face_boxes, face_scores[:, np.newaxis])).astype(np.float32, copy=False)
        body_dets = np.hstack((body_boxes, body_scores[:, np.newaxis])).astype(np.float32, copy=False)

        #keep = py_cpu_nms(dets, args.nms_threshold)
        face_keep = nms(face_dets, args.nms_threshold,force_cpu=args.cpu)
        body_keep = nms(body_dets, args.nms_threshold,force_cpu=args.cpu)

        face_dets = face_dets[face_keep, :]
        body_dets = body_dets[body_keep, :]

        # keep top-K faster NMS
        face_dets = face_dets[:args.keep_top_k, :]
        body_dets = body_dets[:args.keep_top_k, :]
        t3 =time.time()
        print("postprocess", t3-t2)
        #_t['misc'].toc()
       
        tt_01 = time.time()
        # save dets
        if args.outformat == 'JSON':
            face_[os.path.basename(image_path)] =  []
            body_[os.path.basename(image_path)] =  []
         
        if args.outformat == "FDDB":
            fwf.write('{:s}\n{:.0f}\n'.format(img_name, face_dets.shape[0]))
            fwb.write('{:s}\n{:.0f}\n'.format(img_name, body_dets.shape[0]))

        if args.outformat == 'WIDER':
            fw.write('{:s} {:.0f}'.format(img_name, face_dets.shape[0] + body_dets.shape[0]))
    
        if args.outformat == 'VOC':
            filename = os.path.join(anno_dir, '%s.xml') % (os.path.splitext(os.path.basename(idx))[0])
            foc = open(filename, "w")
            foc.write(head)

        print("face dets shape:", face_dets.shape)
        for k in range(face_dets.shape[0]):
            nbox = face_dets[k, :]
            
            if args.outformat == 'JSON':
                face_[os.path.basename(image_path)].append(list(map(float, nbox)))

            if args.outformat == "FDDB":
                fwf.write('{:f} {:f} {:f} {:f} {:f}\n'.format(nbox[0], nbox[1], nbox[2], nbox[3], nbox[4]))

            if args.outformat == 'WIDER':
                fw.write(' {:f} {:f} {:f} {:f} {:f} {:.0f}'.format(nbox[0], nbox[1], nbox[2], nbox[3], nbox[4], 1))

            if args.outformat == 'VOC':
                foc.write(objstr % ('face', int(nbox[0]), int(nbox[1]), int(nbox[2]), int(nbox[3]), nbox[4]))


        print("body dets shape:", body_dets.shape)
        for k in range(body_dets.shape[0]):
            nbox = body_dets[k, :]
            
            if args.outformat == 'JSON':
                body_[os.path.basename(image_path)].append(list(map(float, nbox)))

            if args.outformat == "FDDB":
                fwb.write('{:f} {:f} {:f} {:f} {:f}\n'.format(nbox[0], nbox[1], nbox[2], nbox[3], nbox[4]))

            if args.outformat == 'WIDER':
                fw.write(' {:f} {:f} {:f} {:f} {:f} {:.0f}'.format(nbox[0], nbox[1], nbox[2], nbox[3], nbox[4], 2))

            if args.outformat == 'VOC':
                foc.write(objstr % ('body', int(nbox[0]), int(nbox[1]), int(nbox[2]), int(nbox[3]), nbox[4]))

        if args.outformat == 'VOC':
            foc.write(tail)
            foc.close()
        if args.outformat == 'WIDER':
            fw.write('\n')

        tt_02 = time.time()
        print("save_box_time:", tt_02-tt_01)
        tt_1 = time.time()
        print("total_time:", tt_1-tt_0, "\n")
    if args.outformat == 'JSON':
        json.dump(face_, codecs.open(os.path.join(args.save_folder, os.path.splitext(os.path.basename(args.model))[0]) + '_face_dets.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
        json.dump(body_, codecs.open(os.path.join(args.save_folder, os.path.splitext(os.path.basename(args.model))[0]) + '_body_dets.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
        fwf.close()
        fwb.close()
