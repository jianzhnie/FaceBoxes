# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2, sys, os
import argparse


parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--confidence_threshold', default=0.09, type=float, help='confidence_threshold')
parser.add_argument('--iou_thresh', default=0.3, type=float, help='iou_thresh')
parser.add_argument('--face_thresh', default=0.1, type=float, help='face_threshold')
parser.add_argument('--body_thresh', default=0.1, type=float, help='body_threshold')
parser.add_argument('--test_result',  type=str, help='detected results in WIDER format')
parser.add_argument('--groundtruth',  type=str, help='ground truth file in WIDER format')
args = parser.parse_args()

labels = ['0_all', 'face', 'body']

class MatchBox():

	def __init__(self, iou_threshold=0.3):
		
		self.iou_thresh = iou_threshold

	def match(self, resultsfile, groundtruthfile, obj):
		"""
		匹配检测框和标注框, 为每一个检测框得到一个最大交并比   
		:param resultsfile: 包含检测结果的.txt文件
		:param groundtruthfile: 包含标准答案的.txt文件
		:param show_images: 是否显示图片
		:return maxiou_confidence: np.array, 存放所有检测框对应的最大交并比和置信度
		:return num_detectedbox: int, 检测框的总数
		:return num_groundtruthbox: int, 标注框的总数
		"""

		res_label_dict = Dataloader(resultsfile).loadResult()
		gt_label_dict  = Dataloader(groundtruthfile).loadGTruth()

		assert len(res_label_dict.keys()) == len(gt_label_dict.keys()), "数量不匹配: 标准答案中图片数量为%d, 而检测结果中图片数量为%d" % (
		len(res_label_dict.keys()), len(gt_label_dict.keys()))
		
		maxiou_conf_res = np.array([])
		maxiou_conf_gt = np.array([])

		resdict, res_num = res_label_dict[str(labels.index(obj))]
		gtdict, gt_num  = gt_label_dict[str(labels.index(obj))]


		for img in resdict.keys():
		
			undetbox = []
			falsedetbox = []
			detbox = []

			fname = img # 若需可视化, 修改这里为存放图片的路径
			image = cv2.imread(fname)

			resboxes =  resdict[img]
			gtboxes = gtdict[img]
			for resb in resboxes: # 对于一张图片中的每一个检测框

				iou_array = np.array([0])
				detectedbox = list(map(float, resb[0:5]))
				confidence = detectedbox[-1]
				x_min, y_min, x_max, y_max = int(detectedbox[0]), int(detectedbox[1]), int(detectedbox[2]), int(detectedbox[3])

				gtboxes = gtdict[img]  #groundtruth[results[i][0]]
				for gtb in gtboxes: # 去匹配这张图片中的每一个标注框
					groundtruthbox = list(map(int, gtb[0:4]))
					iou = self.cal_IoU(detectedbox, groundtruthbox)
					iou_array = np.append(iou_array, iou) # 得到一个交并比的数组

				maxiou = np.max(iou_array) #最大交并比
				maxiou_conf_res = np.append(maxiou_conf_res, [maxiou, confidence])


			for gtb in gtboxes: # 对于一张图片中的每一个标注框

				iou_array = np.array([0])
				groundtruthbox = list(map(int, gtb[0:4]))
				x_min, y_min, x_max, y_max = groundtruthbox[0], groundtruthbox[1], groundtruthbox[2], groundtruthbox[3]
				conf = 0

				for resb in resboxes: # 对于一张图片中的每一个检测框
					detectedbox = list(map(float, resb[0:5]))
					confidence = detectedbox[-1]
					iou = self.cal_IoU(detectedbox, groundtruthbox)
					iou_array = np.append(iou_array, iou) # 得到一个交并比的数组
					if np.max(iou_array) == iou:
						conf = confidence

				maxiou = np.max(iou_array) #最大交并比
				maxiou_conf_gt = np.append(maxiou_conf_gt, [maxiou, conf])
				
		maxiou_conf_res = maxiou_conf_res.reshape(-1, 2)
		maxiou_conf_res = maxiou_conf_res[np.argsort(-maxiou_conf_res[:, 1])] # 按置信度从大到小排序
		tf_conf_res = self.thres(maxiou_conf_res, iou_thresh=self.iou_thresh)

		maxiou_conf_gt = maxiou_conf_gt.reshape(-1, 2)
		maxiou_conf_gt = maxiou_conf_gt[np.argsort(-maxiou_conf_gt[:, 1])] # 按置信度从大到小排序
		tf_conf_gt = self.thres(maxiou_conf_gt, iou_thresh=self.iou_thresh)


		return tf_conf_res, res_num, tf_conf_gt, gt_num


	def plot(self, res_conf, gt_num, gt_conf, conf_thresh=0.21):
		"""
		从上到下截取tf_confidence, 计算并画图
		:param tf_confidence: np.array, 存放所有检测框对应的tp或fp和置信度
		:param num_groundtruthbox: int, 标注框的总数
		"""
		recall_list = []
		precision_list = []
		AP = 0

		print("gt_num:", gt_num, "res_num:", len(res_conf))
		length = max(len(res_conf), len(gt_conf))

		for num in range(length):
			arr = res_conf[:(num + 1), 0] # 截取, 注意要加1
			tp = np.sum(arr)
			fp = np.sum(arr == 0)

			conf_t = res_conf[num, 1]
			real_c = len(gt_conf[gt_conf[:,1] >=conf_t])

			recall = real_c / gt_num
			precision = tp / (tp + fp)
			recall_list.append(recall)
			precision_list.append(precision)
			
		
		for i in range(len(precision_list) - 1):
			tm = precision_list[i+1] * (recall_list[i+1] - recall_list[i])
			AP += tm
		
		return AP, recall, precision
		
	def getindex(self, resultfile, groundtruthfile):

		result = {}
		for i in range(1,len(labels)):
			tf_conf_res, res_num, tf_conf_gt, gt_num = self.match(resultfile, groundtruthfile, labels[i])
			ap, recall, precision = self.plot(tf_conf_res, gt_num, tf_conf_gt)
			result[labels[i]] = [ap, recall, precision]

		return result

	def thres(self, maxiou_confidence, iou_thresh=0.4):
		"""
		将大于阈值的最大交并比记为1, 反正记为0
		:param maxiou_confidence: np.array, 存放所有检测框对应的最大交并比和置信度
		:param threshold: 阈值
		:return tf_confidence: np.array, 存放所有检测框对应的tp或fp和置信度
		"""
		maxious = maxiou_confidence[:, 0]
		confidences = maxiou_confidence[:, 1]
		true_or_flase = (maxious > iou_thresh)
		tf_confidence = np.array([true_or_flase, confidences])
		tf_confidence = tf_confidence.T
		tf_confidence = tf_confidence[np.argsort(-tf_confidence[:, 1])]
		return tf_confidence


	def cal_IoU(self, detectedbox, groundtruthbox):
		"""
		计算两个水平竖直的矩形的交并比
		:param detectedbox: list, [leftx_det, topy_det, width_det, height_det, confidence]
		:param groundtruthbox: list, [leftx_gt, topy_gt, width_gt, height_gt, 1]
		:return iou: 交并比
		"""
		leftx_det, topy_det, rigthx_det, downy_det, _  = detectedbox
		leftx_gt, topy_gt, rightx_gt, downy_gt = groundtruthbox

		width_det, height_det = rigthx_det-leftx_det, downy_det-topy_det
		width_gt, height_gt = rightx_gt-leftx_gt, downy_gt-topy_gt
		
		centerx_det = leftx_det + width_det / 2
		centerx_gt = leftx_gt + width_gt / 2
		centery_det = topy_det + height_det / 2
		centery_gt = topy_gt + height_gt / 2

		distancex = abs(centerx_det - centerx_gt) - (width_det + width_gt) / 2
		distancey = abs(centery_det - centery_gt) - (height_det + height_gt) / 2

		if distancex <= 0 and distancey <= 0:
			intersection = distancex * distancey
			union = width_det * height_det + width_gt * height_gt - intersection
			iou = intersection / union
			#print(iou)
			return iou
		else:
			return 0


class Dataloader():
	
	def __init__(self, txtfile):
		self.file = txtfile
		txtfile = open(self.file, 'r')
		self.lines = txtfile.readlines() # 一次性全部读取, 得到一个list
		txtfile.close()

	def loadResult(self):
		'''
		:param txtfile: 读入的.txt文件, 格式要求与FDDB相同
		:return imagelist: list, 每张图片的信息单独为一行, 第一列是图片名称, 第二列是人脸个数, 后面的列均为列表, 包含4个矩形坐标和1个分数
		:return num_allboxes: int, 矩形框的总个数
		:return label_list [[all, boxnum], [label_1, boxnum], [label_2, boxnum], ...], 元素个数取决于检测到标签的个数, label_1是一个字典
		'''
		lines = [mk.strip().split() for mk in self.lines]

		labels = []
		res_labels = {}

		for mk in lines:
			box = []
			k = 0
			while k < int(mk[1])*6:
				box.append(mk[k+2:k+8])
				if mk[k+7] not in labels:
					labels.append(mk[k+7])
				k += 6

		for lb in labels:
			nbox = 0
			tmpdict = {}
			for mk in lines:
				box = []
				k = 0
				while k < int(mk[1])*6:
					if mk[k+7] == lb:
						box.append(mk[k+2:k+8])
						nbox += 1
					k += 6
				tmpdict[os.path.basename(mk[0])] = box
			res_labels[lb] = [tmpdict, nbox]

		return res_labels

	def loadGTruth(self):
		'''
		:param txtfile: 读入的.txt文件, 格式要求与WIDER相同
		:return imagelist: list, 每张图片的信息单独为一行, 第一列是图片名称, 第二列是人脸个数, 后面的列均为列表, 包含4个矩形坐标和1个分数
		:return num_allboxes: int, 矩形框的总个数
		'''
		lines = [mk.strip().split() for mk in self.lines]
 
		labels = []
		gt_labels = {}

		for mk in lines:
			box = []
			k = 0
			while k < int(mk[1])*5:
				box.append(mk[k+2:k+7])
				if mk[k+6] not in labels:
					labels.append(mk[k+6])
				k += 5
		for lb in labels:
			nbox = 0
			tmpdict = {}
			for mk in lines:
				box = []
				k = 0
				while k < int(mk[1])*5:
					if mk[k+6] == lb:
						box.append(mk[k+2:k+7])
						nbox += 1
					k += 5
				tmpdict[os.path.basename(mk[0])] = box
			gt_labels[lb] = [tmpdict, nbox]

		return gt_labels


if __name__=='__main__':
	
	"""
	读取包含检测结果和标准答案的两个.txt文件, 画出ROC曲线和PR曲线
	:param resultsfile: 包含检测结果的.txt文件
	:param groundtruthfile: 包含标准答案的.txt文件
	:param show_images: 是否显示图片, 若需可视化, 需修改Calculate.match中的代码, 找到存放图片的路径
	:param threshold: IoU阈值
	"""
	
	result_file = args.test_result
	gt_file = args.groundtruth

	Index = MatchBox(iou_threshold=args.iou_thresh)
	result = Index.getindex(result_file, gt_file)
	print(result)
