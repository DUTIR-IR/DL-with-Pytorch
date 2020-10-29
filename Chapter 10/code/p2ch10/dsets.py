import copy
import csv
import functools
import glob
import os
import sys

sys.path.append('until')

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch10_raw')

#p259
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple', 
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

#p260-p261
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    
    #处理annotations.csv得到 结节直径、结节中心坐标信息
    diameter_dict = {}
    with open('data/part2/luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    candidateInfo_list = []
    
    #Unifying our annotation and candidate data
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            #如果数据现在不在磁盘上，先跳过，先处理目前在磁盘上的数据子集
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            
            #判断是否是同一结节
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

class Ct:
    #p263
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0]
        
        #加载ct图像，转化为numpy数组
        ct_mhd = sitk.ReadImage(mhd_path)#除了传入的.mhd文件之外，sitk.ReadImage 还会隐式使用.raw文件
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)#将读取出来的图像信息用像素值表示出来

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)#将ct_a数组中的元素值限定在(-1000, 1000)

        self.series_uid = series_uid
        self.hu_a = ct_a
        
        #对下面三个参数的详细解释 https://blog.csdn.net/qq_39071739/article/details/106487597?utm_medium=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.wap_blog_relevant_pic2&depth_1-utm_source=distribute.wap_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.wap_blog_relevant_pic2  
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())#读取图像的原点信息
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())#图像各维度上像素之间的物理距离（单位一般为mm)
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)#采用方向余弦矩阵，这里是指图像本身坐标系相对于世界坐标系（固定不动的）的角度余弦
    
    #p271
    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )#将候选结节坐标（xyz）转换为irc坐标

        slice_list = []
        
        #裁剪整个ct图像，提取出结节图像
        for axis, center_val in enumerate(center_irc):#center_irc是一个元组（i坐标的值，r的值，c的值）
            start_ndx = int(round(center_val - width_irc[axis]/2))#width_irc :图像的深度、高度、宽度
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])
            
            #当如上做法导致所裁剪的小区域的起始坐标和结束坐标超出整个ct图像的边界时
            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]#（[s,e],[],[]）

        return ct_chunk, center_irc

#p274
#functools.lru_cache的作用主要是用来做缓存，他能把相对耗时的函数结果进行保存，避免传入相同的参数重复计算
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
#裁剪后的样本数组
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

#得到张量
class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
            ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
            
        #划分训练集和验证集
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))
    #p272
    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        #根据索引ndx获取候选结节信息的元组
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)
        
        #获取裁剪过的小ct区域块 及 结节中心
        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )
        #p273
        candidate_t = torch.from_numpy(candidate_a)#把数组转换成张量，且二者共享内存
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)#增加一个维度 （adds the‘Channel’ dimension.）

        pos_t = torch.tensor([
                not candidateInfo_tup.isNodule_bool,
                candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long,
        )

        return (
            candidate_t,#裁剪过的ct样本数组对应的张量
            pos_t, #标签张量
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )
