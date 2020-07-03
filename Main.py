#coding=utf-8
from enum import Enum
from functools import partial
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5 import QtTest
import os
import cv2

from skimage import io as iio
import numpy as np
import gdal
import time
import copy as cp

import prediction.prediction_tang
import dbUI
import mydb
from labelDialog import *
def opencv_major_version():
    return int(cv2.__version__.split(".")[0])
def get_files(path, type_='file', format_='*'):
    assert type_ in ['file', 'folder']
    name_list = []
    if type_ == 'file':
        name_list = [x for x in os.listdir(path) if any(x.endswith(extension) for extension in format_)]
    elif type_ == 'folder':
        name_list = [x for x in os.listdir(path) if os.path.isdir(path + x)]

    return name_list
class SpinBoxWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(SpinBoxWindow, self).__init__(parent)
        self.resize(200,200)
        layout =QtWidgets.QVBoxLayout()

        self.l1 = QtWidgets.QLabel("appoximate value:")
        self.l1.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.l1)
        self.sp =QtWidgets.QSpinBox()
        self.sp.setMinimum(1)
        self.sp.setMaximum(10)
        self.sp.setValue(1)
        layout.addWidget(self.sp)
        self.sp.valueChanged.connect(self.valuechange)
        wid=QtWidgets.QWidget()
        wid.setLayout(layout)
        self.setCentralWidget(wid)
        self.setWindowTitle("Approx poly dp")
        self.x0=5.0
        self.x1=10.0
        self.y0=0.01
        self.y1=0.001
        self.valuechange()
    def valuechange(self):
        x=float(self.sp.value())
        AnnotationScene.approx_poly_dp_mis=((x-self.x1)*self.y0-self.y1*(x-self.x0))/(self.x0-self.x1)
class GripItem(QtWidgets.QGraphicsPathItem):
    circle = QtGui.QPainterPath()
    circle.addEllipse(QtCore.QRectF(-2, -2, 5, 5))
    square = QtGui.QPainterPath()
    square.addRect(QtCore.QRectF(-5, -5, 10, 10))

    def __init__(self, annotation_item, index):
        super(GripItem, self).__init__()
        self.m_annotation_item = annotation_item
        self.m_index = index
        #self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setPath(GripItem.circle)
        self.setBrush(QtGui.QColor("green"))
        self.setPen(QtGui.QPen(QtGui.QColor("green"), 2))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(11)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def hoverEnterEvent(self, event):
        self.setPath(GripItem.square)
        self.setBrush(QtGui.QColor("red"))
        super(GripItem, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPath(GripItem.circle)
        self.setBrush(QtGui.QColor("green"))
        super(GripItem, self).hoverLeaveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setSelected(False)
        super(GripItem, self).mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.isEnabled():
            self.m_annotation_item.movePoint(self.m_index, value)
        return super(GripItem, self).itemChange(change, value)


class PolygonAnnotation(QtWidgets.QGraphicsPathItem):
    cur_class=0 #which class is in now
    color_table=[]  #[ [class1,(r1,g1,b1,alpha1)]  , [ class2,(r2,g2,b2,alpha2)]]
    def __init__(self,my_scene=None, parent=None):
        assert my_scene!=None #polygon scene can not be none!
        super(PolygonAnnotation, self).__init__(parent)
        self.m_points = [[]]    #存储所有坐标点 [[qpf(x1,y1),qpf(x2,y2),... qpf(xn,yn)], [qpf(x1,y1) , qpf(x2,y2)] ,...]
        #一般而言len(self.m_points，) == 1 然而，当和深度学习结合后，由于需要显示带洞的图，后面会append多几组洞洞坐标.
        self.poly_scene=my_scene    #表明多边形正放在哪个scene上面.
        self.del_instruction=Instructions.Hand_instruction #默认不进入删除模式
        self.my_color= PolygonAnnotation.cur_class  #作为color_table的下标 用于指示画本个多边形的颜色
        #color信息其实就是类别信息.
        self.dock_idx=-1    #表示这是dock第几行 默认从零开始 主要用于在dock中 删除或者修改这行信息
        self.setZValue(10)
        qpen=QtGui.QPen(QtGui.QColor("green"), 1)
        qpen.setCosmetic(True)
        qpen.setWidth(1)
        self.setPen(qpen)

        self.setAcceptHoverEvents(True)

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        self.m_items = []
        #重新上色使得能够保存
    def set_color_brush(self):
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)

        col=PolygonAnnotation.color_table[self.my_color][1]
        self.setPen(QtGui.QPen(QtGui.QColor(0,0,0,0)))
        self.setBrush(QtGui.QColor(col[0], col[1], col[2]))
    #保存之后把空间恢复原样
    def retoring_color_brush(self):
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)

        col=PolygonAnnotation.color_table[self.my_color][1]
        qpen=QtGui.QPen(QtGui.QColor("green"), 2)
        qpen.setCosmetic(True)
        qpen.setWidth(1)
        self.setPen(qpen)
        self.setBrush(QtGui.QColor(col[0], col[1], col[2], col[3]))


    def number_of_points(self):
        return len(self.m_items)
    #input:poi_list we can set a polygon described by this
    #poi_list = [QPointF1, QPointF2, ... QPointFn ]
    def MySetPolygon(self,poi_list):
        for idx,poi in enumerate(poi_list):
            self.m_points[0].append(poi)
            item=GripItem(self,idx)
            self.scene().addItem(item)
            self.m_items.append(item)
            item.setPos(poi)
        path_tmp=self.from_points_2_path(self.m_points)
        self.setPath(path_tmp)
    #input:exterior_data  [  [ (x1,y1),(x2,y2) ,...  ] ,...  [(x1,y1),（x2,y2) , ...]  ]
    #每一组应该分别用 painter addpolygon
    def MySetFatherSonPolygon(self,exterior_data):
        if len(exterior_data[0])<=1:return #当只有1个点或者没有点的时候需要退出
        count=0

        for gno,group in enumerate(exterior_data):
            tmplist=[]
            for poi in group:
                tmplist.append( QtCore.QPointF(poi[0],poi[1]))

                item=GripItem(self,count)
                self.scene().addItem(item)
                self.m_items.append(item)
                item.setPos(QtCore.QPointF(poi[0],poi[1]))
                count+=1
            if gno:
                self.m_points.append(tmplist)
            else:
                self.m_points[0]=cp.deepcopy(tmplist)

        path_tmp=self.from_points_2_path(self.m_points)
        self.setPath(path_tmp)
    def addPoint(self, p):
        self.m_points[0].append(p)
        print(self.m_points[0])
        path_tmp=self.from_points_2_path(self.m_points)
        self.setPath(path_tmp)
        item = GripItem(self, len(self.m_points[0]) - 1)
        self.scene().addItem(item)
        self.m_items.append(item)
        item.setPos(p)
        #item.setParentItem(self)
    #input:qpointf list
    #output:qtgui.qpainterpath
    #function:把多边形的点转为path后可以setpath
    def from_points_2_path(self,points):
        path_tmp=QtGui.QPainterPath()
        for group in points:
            poly=QtGui.QPolygonF(group)
            path_tmp.addPolygon(poly)
            path_tmp.closeSubpath()
        return path_tmp

    def removeLastPoint(self):
        if self.m_points[0]:
            self.m_points[0].pop()
            path_tmp=self.from_points_2_path(self.m_points)
            self.setPath(path_tmp)
            it = self.m_items.pop()
            self.scene().removeItem(it)
            del it

    def movePoint(self, i, p):
        len_sum=0
        gno=-1
        idx=-1
        #print("mouse move in scene detected!")
        #print(self.m_points)
        #print(i)
        for gno,group in enumerate(self.m_points):
            len_sum+=len(group)
        tmp_lensum=len_sum
        len_sum=0
        if 0 <= i < tmp_lensum:
            for gno,group in enumerate(self.m_points):
                if not gno:old_lensum=0
                else:old_lensum=len_sum
                len_sum+=len(group)
                if i+1<=len_sum:
                    idx=i-old_lensum
                    break
            
            assert  gno!=-1 and idx!=-1
            self.m_points[gno][idx] = self.mapFromScene(p)
            #print("move point ",self.m_points[gno][idx])
            path_tmp=self.from_points_2_path(self.m_points)
            self.setPath(path_tmp)

    def move_item(self, index, pos):
        print("new point x y%d %d"%(pos.x(),pos.y()))

        if 0 <= index < len(self.m_items):
            item = self.m_items[index]
            item.setEnabled(False)
            item.setPos(pos)
            item.setEnabled(True)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            count=0
            for group in self.m_points:
                for point in group:
                    self.move_item(count, self.mapToScene(point))
                    count+=1


        return super(PolygonAnnotation, self).itemChange(change, value)

    def hoverEnterEvent(self, event):
        col=PolygonAnnotation.color_table[self.my_color][1]

        self.setBrush(QtGui.QColor(col[0], col[1], col[2], col[3]))
        super(PolygonAnnotation, self).hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        #self.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        super(PolygonAnnotation, self).hoverLeaveEvent(event)
    def mouseDoubleClickEvent(self, event):
        print("double click polygon!")
        ret=self.poly_scene.parent.LabelDialog.popUp()
        if ret is None:return
        self.my_color=ret
        item=self.poly_scene.parent.dock1_listwidget.item(self.dock_idx)
        item.setText(self.poly_scene.parent.classes_name_color_pair[self.my_color][0])
        self.retoring_color_brush()

        super(PolygonAnnotation, self).mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        print("polygon clicked")
        click_x = event.scenePos().x();
        click_y = event.scenePos().y();
        if self.del_instruction == Instructions.Delete_Instruction:
            print("deleting polygon")
            for it in self.m_items:
                self.poly_scene.removeItem(it)
            self.poly_scene.removeItem(self)
            self.poly_scene.parent.dock1_listwidget.takeItem(self.dock_idx)
            for idx,poly in enumerate(Allpoly.all_poly):
                if poly.dock_idx>self.dock_idx:
                    print(poly)
                    poly.dock_idx-=1
            for idx,poly in enumerate(Allpoly.all_poly):
                if poly == self:
                    del Allpoly.all_poly[idx]
                    break
            for it in Allpoly.all_poly:
                it.del_instruction=Instructions.Hand_instruction 
        else:   #now we support function of add point within a line
            k_diff_const = 1e-1   # the k diff we allow
            lspoint = 0;
            for group_no,group in enumerate(self.m_points):
                suc = 0;
                for idx in range(len(group)):
                    lsy = group[idx].y()
                    lsx = group[idx].x()

                    nxy = group[(idx+1)%len(group)].y()
                    nxx = group[(idx+1)%len(group)].x()

                    k1=(nxy - lsy)/(nxx - lsx + 1e-12) 
                    k2 = (click_y - lsy) / (click_x - lsx + 1e-12)
                    
                    minx = min(lsx,nxx)
                    maxx = max(lsx,nxx)

                    miny = min(lsy,nxy)
                    maxy = max(lsy,nxy)

                    if abs(k1-k2)<=abs(k1*2/10) and click_x>=minx and click_x<= maxx and click_y>=miny and click_y<=maxy  :    #we find a potential point of inserting
                        
                        self.m_points[group_no].insert(idx+1,event.scenePos());
                        
                        path_tmp=self.from_points_2_path(self.m_points)
                        self.setPath(path_tmp)
                        item = GripItem(self, lspoint+idx+1)
                        self.scene().addItem(item)
                        self.m_items.insert(lspoint+idx+1,item);
                        for j in range(lspoint+idx+2,len(self.m_items)):    #all points idx below should plus 1
                                self.m_items[j].m_index+=1
                        item.setPos(event.scenePos())
                        
                        suc=1;
                        break;
                    else:continue;
                lspoint+=len(group)
                if suc:break;
                    




        super(PolygonAnnotation, self).mousePressEvent(event)


class Instructions():
    No_Instruction = 1
    Polygon_Instruction = 2
    Contour_Instruction=3  #while users wants to use contour to display contours
    Delete_Instruction = 4  #while deleting a polygon we need to use this
    Hand_instruction=5 #while no delete a polygon we need to use this and maybe in other cases
    Polygon_Finish=6    #while polygon is drawn we need to set this instruction



class Allpoly():
    all_poly=[] #注意这里的下标耦合了dock的下标，也就是这里的元素的下标要和dock的一致
class AnnotationScene(QtWidgets.QGraphicsScene):
    approx_poly_dp_mis=0.01    #表示我们在用flood fill或者深度学习时生成多边形时允许的误差
    def __init__(self, parent=None):
        super(AnnotationScene, self).__init__(parent)
        self.image_item = QtWidgets.QGraphicsPixmapItem()
        self.polygon_item = PolygonAnnotation(self)
        self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.addItem(self.image_item)
        self.parent=parent  #scene的父亲，一般默认是主界面QMainWindow
        self.cv_nir_img=None    #once we load a image, we need to add a [nir r g] image here
        self.cv_img=None    #once we load a image, we need to add a [r g b] image .
        self.file_name=""   #once we load a new file, we need to change its filename.
        self.show_nir=0 #0 表示正在显示RGB 1 表示正在显示近红外信息
        self.low_threshhold=100 #canny low threshold
        self.high_threshold=200 #canny high threshold
        self.all_poly=[]    #insert all polygon
        self.dpl_suffix=[".jpg",".png",".tif",".tiff"]  #深度学习允许后缀
        self.current_instruction = Instructions.Hand_instruction
    def itemDisappearShow(self):    #让所有多边形消失，再按一次X键所有控件显示.
        if self.file_name is "":return
        func=self.itemDisappearShow.__func__
        if not hasattr(func,"flag"):
            func.flag=0

        for poly in Allpoly.all_poly:
            self.removeItem(poly)
            for grip in poly.m_items:
                self.removeItem(grip)
        if func.flag:
            for poly in Allpoly.all_poly:
                self.addItem(poly)
                for grip in poly.m_items:
                    self.addItem(grip)
        func.flag=not func.flag
    #input:full_path 一般为self.file_name目的是得到前一级 folder以及当前的图片的名字
    def get_folder_and_image_name(self,full_path):
        if len(full_path.split("\\"))>1:
            ls = full_path.split("\\")
            return (ls[0],ls[1])
        ls=self.file_name.split("/")
        img_name=ls.pop()
        ls2=img_name.split(".")
        ls2.pop()
        img_name="."
        img_name= img_name.join(ls2)

        folder="/"
        folder=folder.join(ls)
        return (folder,img_name)
    def deep_learning_pth(self):
        if self.file_name=="":
            print("no image loaded before")
            return
        folder,img_name=self.get_folder_and_image_name(self.file_name)
        if not os.path.isdir(folder+"/label/"):os.mkdir(folder+"/label/")
        prediction.prediction_tang.deep_learning_glue(self.file_name,folder+"/label/")
        print("model pth process complete")
        self.deep_learning_show()

    #注意 使用深度学习的label的后缀为jpg,png,tif,tiff
    def deep_learning_show(self):
        print(self.file_name)
        if self.file_name=="":
            print("no image loaded before")
            return
        folder,img_name=self.get_folder_and_image_name(self.file_name)
        actsuf=""
        for suf in self.dpl_suffix:
            print(folder+'/label/'+img_name+'_label'+suf)
            if not os.path.exists(folder+'/label/'+img_name+'_label'+suf):continue
            actsuf=suf
        if actsuf == "":
            print(folder+'/label/'+img_name+'_label'+actsuf)
            print("none exist")
            return
        label_img=folder+'/label/'+img_name+'_label'+actsuf
        label_img=iio.imread(label_img)
        to=len(PolygonAnnotation.color_table)+1
        for idx in range(1,to):
            self.deep_learning_area_append(label_img,idx)


    #从深度学习里面提取
    # label图片的target class的轮廓，并将其轮廓区域送到对应polygon_item
    def deep_learning_area_append(self,label_img,target_class):
        if len(label_img.shape)>=3:label_img=label_img[:,:,0]
        ro,col=label_img.shape
        img=np.zeros(label_img.shape,dtype=np.uint8)
        count=0
        for x in np.nditer(label_img):
            if label_img[count//col][count%col]==target_class:
                img[count//col][count%col]=255
            count+=1
        null_t=np.zeros(img.shape,dtype=np.uint8)
        if opencv_major_version()!=3:
            cont,hierarchy=cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        else:
            __,cont,hierarchy=cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        mis_match=AnnotationScene.approx_poly_dp_mis
        tree=[]
        for i in range(len(cont)):
            tree.append([])
        father=[]
        for i in range(len(cont)):
            if hierarchy[0][i][3]==-1:
                father.append(i)
            else:
                fa=np.uint8(hierarchy[0][i][3])
                tree[fa].append(i)
        for i in father:
            epsilon = mis_match*cv2.arcLength(cont[i],True)
            approx = cv2.approxPolyDP(cont[i],epsilon,True)
            my_data_exchange=[]
            tmp=[]
            for ii in approx:
               tmp.append((ii[0][0],ii[0][1]))
            my_data_exchange.append(tmp)
            for j in tree[i]:
                epsilon = mis_match*cv2.arcLength(cont[j],True)
                approx = cv2.approxPolyDP(cont[j],epsilon,True)
                tmp=[]
                for ii in approx:
                   tmp.append((ii[0][0],ii[0][1]))
                my_data_exchange.append(tmp)
            self.setCurrentInstruction(Instructions.Polygon_Instruction)
            self.polygon_item.my_color=target_class-1
            print("setting target class father %d"%(target_class))
            self.polygon_item.MySetFatherSonPolygon(my_data_exchange)
            self.setCurrentInstruction(Instructions.Polygon_Finish)

    #if we want to display a nir we need to use this
    def Flip_show(self):
        if self.cv_nir_img is None or self.cv_img is None:return
        self.show_nir=not self.show_nir
        if self.show_nir:
            self.load_image(self.cv_nir_img)
        else:
            self.load_image(self.cv_img)
    #convert a opencv img to show in QT
    def load_image(self, cv_img):
        img = cv_img.copy()
        x = img.shape[1]                                                        #获取图像大小
        y = img.shape[0]
        print(img[0,:].nbytes)
        print(x)
        print(y)
        frame =QtGui.QImage(np.uint8(img), x, y,img[0,:].nbytes, QtGui.QImage.Format_RGB888)
        self.image_item.setPixmap(QtGui.QPixmap.fromImage(frame))
        self.setSceneRect(self.image_item.boundingRect())

    def setCurrentInstruction(self, instruction):
        self.current_instruction = instruction
        if instruction == Instructions.Hand_instruction:
            for it in Allpoly.all_poly:
                it.del_instruction=Instructions.Hand_instruction
            return
        elif instruction== Instructions.Delete_Instruction:
            for it in Allpoly.all_poly:
                it.del_instruction=Instructions.Delete_Instruction
            return
        elif instruction== Instructions.Polygon_Instruction:
            for it in Allpoly.all_poly:
                it.del_instruction=Instructions.Hand_instruction

            self.polygon_item = PolygonAnnotation(self)

            print("all poly append")

            print(self.polygon_item)

            self.addItem(self.polygon_item)
            print("add successfully")
        elif instruction==Instructions.Polygon_Finish:
            if len(self.polygon_item.m_points[0])<=1:return    #当没有点输入，我们可以直接退出
            Allpoly.all_poly.append(self.polygon_item)

            dock1_idx=self.parent.dock1_listwidget.count()
            self.polygon_item.dock_idx=dock1_idx
            
            tmpstr=self.parent.classes_name_color_pair[self.polygon_item.my_color][0]
            self.polygon_item = PolygonAnnotation(self) #清空
            self.parent.dock1_listwidget.addItem(tmpstr)
        else:pass
    #ref:https://www.cnblogs.com/anningwang/p/7581545.html
    #input: cnt .where cnt from cv2.findcontours(),the first output contour
    #input: ts_x : x coordinate of the checking point
    #input: ts_y : y coordinate of the checking point
    #output:1 inside
    #output:0 outside
    def judge_point_in_polygon(self,cnts,ts_x,ts_y):
        x_array = []
        y_array = []
        nvert = 0
        for i in range(cnts.shape[0]):
            (x,y) = cnts[i][0]
            x_array.append(x)
            y_array.append(y)
            nvert+=1
        minx = sorted(x_array)[0]
        miny = sorted(y_array)[0]
        max_x = sorted(x_array,reverse = True)[0]
        max_y = sorted(y_array,reverse = True)[0]
        if (ts_x < minx or ts_x > max_x or ts_y < miny or ts_y > max_y):
            return 0
        c = 0
        j = nvert - 1
        for i in range(nvert):
            if ( ((y_array[i]>ts_y) != (y_array[j]>ts_y))and(ts_x < (x_array[j]-x_array[i]) * (ts_y-y_array[i]) / (y_array[j]-y_array[i]) + x_array[i]) ):
                c = not c
            j = i

        return  c
    def seed_point_func(event,x,y,flags,param):
        global glo
        global seed_point
        if event == cv2.EVENT_LBUTTONDOWN:
            glo=1
            seed_point[0]=x
            seed_point[1]=y
    def mousePressEvent(self, event):
        if self.cv_img is None:
            super(AnnotationScene, self).mousePressEvent(event)
            return

        if self.current_instruction == Instructions.Polygon_Instruction:
            self.polygon_item.removeLastPoint()
            self.polygon_item.addPoint(event.scenePos())
            print("add point ")
            # movable element
            self.polygon_item.addPoint(event.scenePos())    #add two point cuz we want to move a point
        elif self.current_instruction == Instructions.Contour_Instruction:

            x=int(event.scenePos().x())
            y=int(event.scenePos().y())

            #cv2.imshow("ori ",self.cv_img)
            print("ori ")
            img = cv2.cvtColor(self.cv_img,cv2.COLOR_BGR2GRAY)
            #cv2.imshow("gray ",img)
            print("gray")
            edges=cv2.Canny(img,self.low_threshhold,self.high_threshold)
            kernel = np.ones((5,5),np.uint8)
            gradient = cv2.dilate(edges,kernel,iterations = 1)
            gradient=255-gradient
            #cv2.imshow("gradient ",gradient)
            #cv2.waitKey(0)
            if opencv_major_version()!=3:
                cnts,_=cv2.findContours(gradient,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            else:
                __,cnts,_=cv2.findContours(gradient,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

            cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
            suc=0
            print("cv part done")
            for idx,cont in enumerate(cntsSorted):
                if self.judge_point_in_polygon(cont,x,y):
                    suc=1
                    break
            if not suc:return
            self.setCurrentInstruction(Instructions.Polygon_Instruction)
            epsilon = AnnotationScene.approx_poly_dp_mis*cv2.arcLength(cntsSorted[idx],True)
            approx = cv2.approxPolyDP(cntsSorted[idx],epsilon,True)
            lis=[]
            for i in range(approx.shape[0]):
                (x,y)=approx[i][0]
                lis.append(QtCore.QPointF(x,y))
            self.polygon_item.MySetPolygon(lis)
            print("add point part done")
            self.setCurrentInstruction(Instructions.Polygon_Finish)

        else:pass
        super(AnnotationScene, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.current_instruction == Instructions.Polygon_Instruction:
            #print("move point %d"%(self.polygon_item.number_of_points()-1))
            self.polygon_item.movePoint(self.polygon_item.number_of_points()-1, event.scenePos())
            if self.polygon_item.number_of_points():
                self.polygon_item.m_items[-1].setPos(event.scenePos())
        super(AnnotationScene, self).mouseMoveEvent(event)


class AnnotationView(QtWidgets.QGraphicsView):
    factor = 2.0

    def __init__(self, parent=None):
        super(AnnotationView, self).__init__(parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        QtWidgets.QShortcut(QtGui.QKeySequence.ZoomIn, self, activated=self.zoomIn)
        QtWidgets.QShortcut(QtGui.QKeySequence.ZoomOut, self, activated=self.zoomOut)

    @QtCore.pyqtSlot()
    def zoomIn(self):
        self.zoom(AnnotationView.factor)

    @QtCore.pyqtSlot()
    def zoomOut(self):
        self.zoom(1 / AnnotationView.factor)

    def zoom(self, f):
        self.scale(f, f)
        if self.scene() is not None:
            self.centerOn(self.scene().image_item)
    def wheelEvent(self, event):
        factor = 1.1
        if event.angleDelta().y() < 0:
            factor = 0.9
        view_pos = event.pos()
        scene_pos = self.mapToScene(view_pos)
        self.centerOn(scene_pos)
        self.scale(factor, factor)
        delta = self.mapToScene(view_pos) - self.mapToScene(self.viewport().rect().center())
        self.centerOn(scene_pos - delta)
    

class AnnotationWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(AnnotationWindow, self).__init__(parent)
        login = dbUI.Login()
        if login.exec_() == QtWidgets.QDialog.Accepted:
            self.use_db = True;
        else:self.use_db = False;
                
        self.classes_name_color_pair=[["default",(255,0,0,100)]]     #类别颜色对 [  [class1, (r1,g1,b1,alpha1)],[class2,(r2,g2,b2,alpha2) ...]   ]
        self.cur_class=0    #表明当前所在类别
        self.image_list_all_dir=[]  #表明当前正在访问的图片的路径的全称 例如: [D:/Work/Pet.png,D:/Work/Lake.png ...]
        self.folder=""   #表明当前正在访问的目录 例如     "D:/Work"
        self.image_last_dir=[]  #表明当前正在访问的图片 例如： [Pet.png, Lake.png, ...]
        self.image_poi=0    #表明当前正在访问目录的哪一个图片
        self.dock1=QtWidgets.QDockWidget(u"已标注多边形",self)
        self.dock1_listwidget=QtWidgets.QListWidget()   #代表dock的list
        self.dock1.setWidget(self.dock1_listwidget) #每次增删多边形都要对这个进行操作.
        self.dock1_listwidget.itemClicked.connect(self.brush_item)
        self.dock1_ls_poly=None #重新渲染
        self.LabelDialog=None   #用于重新选择多边形的类别.
        self.progress_bar=QtWidgets.QProgressBar(None)
        self.progress_bar.setGeometry(500, 500, 300, 50)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.ccls_action=[] #存放当前类别.
        #self.progress_bar.show()
        self.progress_bar.hide()

        self.m_view = AnnotationView()
        self.m_scene = AnnotationScene(self)

        self.allBandImage=None
        self.bandList=[]
        self.m_view.setScene(self.m_scene)
        self.setCentralWidget(self.m_view)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea,self.dock1)
        self.read_classes_and_colors()  #添加类别和颜色信息
        self.createLabelDialod()#这时候可以生成改变类别信息的label
        self.create_menus()


        QtWidgets.QShortcut(QtCore.Qt.Key_X,self,activated=self.m_scene.itemDisappearShow)
        QtWidgets.QShortcut(QtCore.Qt.Key_Escape, self, activated=partial(self.m_scene.setCurrentInstruction, Instructions.Polygon_Finish))
        QtWidgets.QShortcut(QtCore.Qt.Key_Left, self, activated=partial (self.next_image,-1))
        QtWidgets.QShortcut(QtCore.Qt.Key_Right, self, activated=partial(self.next_image,1))
        QtWidgets.QShortcut(QtCore.Qt.Key_C, self, activated=self.m_scene.Flip_show)


    def createLabelDialod(self):
        out=[stringBox[0] for stringBox in self.classes_name_color_pair]
        self.LabelDialog=LabelDialog(parent=None,listItem=out)

    def brush_item(self):
        idx=self.dock1_listwidget.currentRow()
        if self.dock1_ls_poly is not None and self.dock1_ls_poly in Allpoly.all_poly:
            qpen=QtGui.QPen(QtGui.QColor("green"), 2)
            qpen.setCosmetic(True)
            qpen.setWidth(1)
            self.dock1_ls_poly.setPen(qpen)
        if Allpoly.all_poly[idx] == self.dock1_ls_poly:
            qpen=QtGui.QPen(QtGui.QColor("green"), 2)
            qpen.setCosmetic(True)
            qpen.setWidth(1)
            self.dock1_ls_poly.setPen(qpen)
            return;

        
        qpen=QtGui.QPen(QtGui.QColor("red"), 10)
        qpen.setCosmetic(True)
        Allpoly.all_poly[idx].setPen(qpen)
        self.dock1_ls_poly= Allpoly.all_poly[idx]



    def next_image(self,dir):
        if dir == 1:
            if self.image_poi == len(self.image_list_all_dir)-1:return
        else:
            if not self.image_poi:return
        self.image_poi+=dir


        print(self.image_list_all_dir[self.image_poi])
        self.load_image(self.image_list_all_dir[self.image_poi])
    def prev_image(self):
        pass
    def read_classes_and_colors(self):
        with open("classes.txt","rb") as file:
            count=0
            for line in file:
                line=line.decode("utf-8") 
                if  not count:self.classes_name_color_pair=[]
                self.classes_name_color_pair.append([line])
                count+=1
        count=0
        with open("color.txt","r") as file:
            for line in file:

                t=line.split(",")
                color=[]
                for i in t:
                    assert (int(i)<=255)
                    color.append(int(i))
                color=tuple(color)
                assert(len(color)==4)   #注意颜色总共有4个分量 rgb alpha其中alpha表示透明度
                assert(count<len(self.classes_name_color_pair)) #注意类别颜色一对一
                self.classes_name_color_pair[count].append(color)
                count+=1
        PolygonAnnotation.color_table=self.classes_name_color_pair


    def create_menus(self):

        OpenFileAct = QtWidgets.QAction(QtGui.QIcon('./resources/icons/open.png'), 'Ctrl+O', self)
        
        OpenFileAct.setIconText("Open a file")
        OpenFileAct.setShortcut('Ctrl+O')
        OpenFileAct.setStatusTip('Ctrl+O')
        OpenFileAct.triggered.connect(self.load_image)

        OpendirAct = QtWidgets.QAction(QtGui.QIcon('./resources/icons/open.png'), 'Ctrl+D', self)   
        OpendirAct.setIconText("Open directory")
        OpendirAct.setShortcut('Ctrl+D')
        OpendirAct.setStatusTip('Open directory')
        OpendirAct.triggered.connect(self.load_file_image)

        SaveAct = QtWidgets.QAction(QtGui.QIcon('./resources/icons/save.png'), 'Ctrl+S', self)   
        SaveAct.setIconText("Save")
        SaveAct.setStatusTip('Save')
        SaveAct.triggered.connect(self.save_image)

        PolyAct = QtWidgets.QAction(QtGui.QIcon('./resources/icons/objects.png'), 'Ctrl+N', self)   
        PolyAct.setIconText("New polygon")
        PolyAct.setShortcut('Ctrl+N')
        PolyAct.setStatusTip('New polygon')
        PolyAct.triggered.connect(partial(self.m_scene.setCurrentInstruction, Instructions.Polygon_Instruction))

        PolyDeleteAct = QtWidgets.QAction(QtGui.QIcon('./resources/icons/cancel.png'), 'Ctrl+R', self)   
        PolyDeleteAct.setIconText("Delete a polygon")
        PolyDeleteAct.setShortcut('Ctrl+R')
        PolyDeleteAct.setStatusTip('Delete a polygon')
        PolyDeleteAct.triggered.connect(partial(self.m_scene.setCurrentInstruction, Instructions.Delete_Instruction))


        PolyHandAct = QtWidgets.QAction(QtGui.QIcon('./resources/icons/handicon.png'), 'Ctrl+H', self)   
        PolyHandAct.setIconText("Hand mode")
        PolyHandAct.setShortcut('Ctrl+H')
        PolyHandAct.setStatusTip('Hand mode')
        PolyHandAct.triggered.connect(partial(self.m_scene.setCurrentInstruction, Instructions.Hand_instruction))



        toolbar_speed_dial = QtWidgets.QToolBar('Left menu')
        toolbar_speed_dial.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        toolbar_speed_dial.setIconSize(QtCore.QSize(50,50))
        
        self.addToolBar(QtCore.Qt.LeftToolBarArea, toolbar_speed_dial)
        
        toolbar_speed_dial.addAction(OpenFileAct)
        toolbar_speed_dial.addAction(OpendirAct)
        toolbar_speed_dial.addAction(SaveAct)
        toolbar_speed_dial.addSeparator()
        toolbar_speed_dial.addAction(PolyAct)
        toolbar_speed_dial.addAction(PolyDeleteAct)
        toolbar_speed_dial.addAction(PolyHandAct)

        menu_file = self.menuBar().addMenu("File")
        load_image_action = menu_file.addAction("&Load Image")
        load_file_image_action = menu_file.addAction("&Load File Image")
        load_file_image_action.triggered.connect(self.load_file_image)

        save_action=menu_file.addAction("&Save Image")
        save_action.triggered.connect(self.save_image)
        QtWidgets.QShortcut(QtGui.QKeySequence.Save, self, activated=self.save_image)
        load_image_action.triggered.connect(self.load_image)

        menu_instructions = self.menuBar().addMenu("Intructions")
        polygon_action = menu_instructions.addAction("Polygon")
        polygon_action.triggered.connect(partial(self.m_scene.setCurrentInstruction, Instructions.Polygon_Instruction))
        delete_action = menu_instructions.addAction("Delete Polygon")
        delete_action.triggered.connect(partial(self.m_scene.setCurrentInstruction, Instructions.Delete_Instruction))

        contour_action = menu_instructions.addAction("Magic Pen")
        contour_action.triggered.connect(partial(self.m_scene.setCurrentInstruction, Instructions.Contour_Instruction))
        dpl= menu_instructions.addAction("Deep learning from label image")
        dpl.triggered.connect(self.m_scene.deep_learning_show)

        deep_learning_action=menu_instructions.addAction("Deep learning from pth")
        deep_learning_action.triggered.connect(self.m_scene.deep_learning_pth)

        restoring_action=menu_instructions.addAction("Hand Mode")
        restoring_action.triggered.connect(partial(self.m_scene.setCurrentInstruction, Instructions.Hand_instruction))




        overload_test=self.menuBar().addMenu("Setting")
        save_img_memo_ts=overload_test.addAction("polyApproximate")
        bandSwap=overload_test.addAction("BandSwap")
        bandSwap.triggered.connect(self.bandSwap)
        save_img_memo_ts.triggered.connect(self.polyApproximate)

        ccls=self.menuBar().addMenu("Current Class")
        
        for idx,i in enumerate(self.classes_name_color_pair):

            self.ccls_action.append(ccls.addAction(i[0]))
            
            self.ccls_action[-1].triggered.connect(partial(self.handle_cur_class,idx))
            

        db_instructions = self.menuBar().addMenu("Database")
        sc = db_instructions.addAction("Scan")
        sc.triggered.connect(self.db_scan)
        sch = db_instructions.addAction("Search")
        sch.triggered.connect(self.db_search)
        drt = db_instructions.addAction("Drop table")
        drt.triggered.connect(self.db_drop_table)
        if not self.use_db:
            sc.setEnabled(False)
            sch.setEnabled(False)
            drt.setEnabled(False)
    
        
    @QtCore.pyqtSlot()
    def bandSwap(self):
        if self.allBandImage is None:return
        tmp=self.allBandImage.copy()
        for idx,ele in enumerate(self.bandList):
            ele=int(ele)
            tmp[:,:,ele]=self.allBandImage[:,:,idx].copy()
        dialog=bandOption(self.bandList)
        ret=dialog.popUp()
        if ret is None:return
        self.bandList=ret.copy()
        for idx,ele in enumerate(self.bandList):
            self.allBandImage[:,:,idx]=tmp[:,:,ele]
        self.m_scene.load_image(self.allBandImage[:,:,0:3])
    @QtCore.pyqtSlot()
    def db_scan(self):
        try:
            folder_name =QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Folder","D:\\WORK\\UESTC\\",
                                           QtWidgets.QFileDialog.ShowDirsOnly
                                           | QtWidgets.QFileDialog.DontResolveSymlinks
                          )
        except Exception as e:
            print(e)
            return
        if folder_name is "":return
        try:
            dbUI.scan_and_insert_to_table(folder_name)
        except Exception as e:
            print(e);
            return
    @QtCore.pyqtSlot()
    def db_drop_table(self):
        drop_window = dbUI.drop_table_window()
        drop_window.exec_()

    @QtCore.pyqtSlot()
    def db_search(self):
        lines=mydb.singleton_data_base.select_count_lines()
        if lines == -1:return
        swindow = dbUI.search_window()
        if swindow.exec_() == QtWidgets.QDialog.Accepted:
            res_win = dbUI.select_result_window(swindow.parameter_dict);
            if res_win.exec_()  == QtWidgets.QDialog.Accepted:
                if res_win.ret_data == -1:return
                if not len(res_win.ret_data):return
                self.image_list_all_dir = []
                self.image_last_dir=[]
                self.image_poi=0
                self.folder = None
                for ele in res_win.ret_data:
                    self.image_list_all_dir.append(ele[0])
                    self.image_last_dir.append(ele[0].split("\\")[-1])

                self.load_image(self.image_list_all_dir[self.image_poi])
            else:return;

            
        
    @QtCore.pyqtSlot()
    def polyApproximate(self):
        sp=SpinBoxWindow(self)
        sp.show()

    @QtCore.pyqtSlot()
    def load_file_image(self):
        try:
            folder_name =QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Folder","D:\\WORK\\UESTC\\",
                                           QtWidgets.QFileDialog.ShowDirsOnly
                                           | QtWidgets.QFileDialog.DontResolveSymlinks
                          )
        except Exception as e:

            print(e)
            return
        if folder_name is "":return
        print("opening folder")
        print(folder_name)
        image_names = get_files(folder_name, format_=['jpg', 'png', 'bmp','tif','tiff'])
        print(image_names)
        if not len(image_names):return
        self.image_list_all_dir = []
        self.image_last_dir=[]
        self.folder=folder_name
        self.image_poi=0
        for image_name in image_names:
            self.image_list_all_dir.append(folder_name + '/' + image_name)
            self.image_last_dir.append(image_name)

        self.load_image(self.image_list_all_dir[self.image_poi])



    @QtCore.pyqtSlot()
    def no_delete_command(self):
        self.m_scene.setCurrentInstruction(Instructions.Hand_instruction)
        for it in Allpoly.all_poly:
            it.del_instruction=Instructions.Hand_instruction

    @QtCore.pyqtSlot()
    def delete_command(self):
        self.m_scene.setCurrentInstruction(Instructions.Hand_instruction)
        for it in Allpoly.all_poly:
            it.del_instruction=Instructions.Delete_Instruction
            print(it.del_instruction)


    @QtCore.pyqtSlot()
    def save_memo_ts(self):
        import time
        for i in range(1000):
            self.save_image()
            time.sleep(0.001)

    def handle_cur_class(self,idx=0):
        for i in range(len(self.ccls_action)):
            if i!=idx:
                self.ccls_action[i].setShortcut('')
            else:
                self.ccls_action[i].setShortcut('<')
        self.cur_class=idx
        PolygonAnnotation.cur_class=self.cur_class
    def onCountChanged(self,val):
        self.progress_bar.setValue(val)
    def QImageToCvMat(self,incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        #incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        #incomingImage=incomingImage.convertToFormat(QtGui.QImage.Format_RGB32)
        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr
    @QtCore.pyqtSlot()
    def save_image(self):
        self.progress_bar.show()
        calc = External()
        calc.countChanged.connect(self.onCountChanged)
        calc.start()
        area=self.m_scene.sceneRect()
        if int(area.height())==0 or int(area.width())==0:return 0
        img=np.zeros((int(area.height()),int(area.width()),3),dtype=np.uint8)
        x=img.shape[1]
        y=img.shape[0]
        Qimg =QtGui.QImage(np.uint8(img), x, y,img[0,:].nbytes,QtGui.QImage.Format_RGB888)
        #print(Qimg.save("tmp.png"))
        pixmap = QtGui.QPixmap.fromImage(Qimg)
        pixmap_item=QtWidgets.QGraphicsPixmapItem()
        pixmap_item.setPixmap(pixmap)


        painter=QtGui.QPainter(Qimg)
        painter.setRenderHint( QtGui.QPainter.NonCosmeticDefaultPen)
        save_scene=QtWidgets.QGraphicsScene()
        save_scene.addItem(pixmap_item)
        save_scene.setSceneRect(pixmap_item.boundingRect())
        for item in Allpoly.all_poly:
            item.set_color_brush()
            save_scene.addItem(item)
        # Render the region of interest to the QImage.
        save_scene.render(painter)
        painter.end()
        if not os.path.isdir(self.folder+'/'+'mask'):os.mkdir(self.folder+'/'+'mask')
        # Save the image to a file.
        img_name=self.image_last_dir[self.image_poi]
        ls=img_name.split(".")
        ls.pop()
        tmpstr="."
        tmpstr=tmpstr.join(ls)
        tmpstr=tmpstr+"_mask.tif"
        print(self.folder+'/'+'mask'+'/'+tmpstr)
        if not os.path.isdir(self.folder+"/mask/"):os.mkdir(self.folder+"/mask/")
        img_name=self.folder+'/'+'mask'+'/'+tmpstr

        Qimg.save(img_name)
        npqimg=Qimg.convertToFormat(QtGui.QImage.Format_RGB32)
        nparray=self.QImageToCvMat(npqimg)
        nparray=nparray[:,:,0:3]    #the last is ff we choose to omit it
        #nparray=cv2.cvtColor(nparray,cv2.COLOR_BGR2RGB)

        #[ [class1,(r1,g1,b1,alpha1)]  , [ class2,(r2,g2,b2,alpha2)]]
        gray_mask=np.zeros(nparray.shape,dtype=np.uint8)
        for idx,cls in enumerate(PolygonAnnotation.color_table):
            r=cls[1][0]
            g=cls[1][1]
            b=cls[1][2]
            loc=np.where(((nparray[:,:,2]==r) & (nparray[:,:,1]==g) & (nparray[:,:,0]==b)))
            gray_mask[loc[0],loc[1]]=idx+1

        if not os.path.isdir(self.folder+'/'+'gray_mask'):os.mkdir(self.folder+'/'+'gray_mask')
        gray_img_name=self.folder+'/'+'gray_mask'+'/'+tmpstr
        cv2.imencode('.tif', gray_mask)[1].tofile(gray_img_name)

        for item in Allpoly.all_poly:
            item.retoring_color_brush()
            self.m_scene.addItem(item)

        #save txt file
        ls=img_name.split(".")
        ls.pop()
        ls.append("txt")
        tmpstr="."
        txt_name=tmpstr.join(ls)


        with open(txt_name,"w+") as file:
            #写文件的格式如下：
            #point (x1,y1) (x2,y2)  ... (xn,yn) （通过空格表示间隔,注意存放必须按照多边形的顶点的顺时针或者逆时针 顺序存储！）
            #class x （x表示这个多边形所属类别）

            for poly in Allpoly.all_poly:
                file.write("point")
                tempsum=[0]
                tmps=0
                for gno,ele in enumerate(poly.m_points):
                    tmps+=len(ele)
                    tempsum.append(tmps)
                lsgno=0
                print(tempsum)
                for item in poly.m_items:
                    idx=item.m_index

                    gidx=self.binary_search(tempsum,0,len(tempsum),idx+1)
                    #print("binary search complete idx:%d gidx:%d"%(idx+1,gidx))
                    gidx-=1
                    if gidx!=lsgno:
                        file.write("$")
                        lsgno=gidx
                    else:
                        file.write(" ")
                    poi=item.scenePos()

                    (x,y)=(poi.x(),poi.y())
                    file.write("(")
                    file.write(str(x))
                    file.write(",")
                    file.write(str(y))
                    file.write(")")
                file.write("\n")
                file.write("class ")
                file.write(str(poly.my_color))
                file.write("\n")

        self.progress_bar.setValue(100)
        QtTest.QTest.qWait(500)
        self.progress_bar.hide()

    def binary_search(self,lis,x,y,goal):
        while x<y:

            m=x+(y-x)//2
            if not m:break
            if lis[m]>goal:
                y=m
            elif goal>lis[m-1] and goal<=lis[m]:return m
            else :x=m+1
        return y


    @QtCore.pyqtSlot()
    def load_image(self,filename=None):
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                "Open Image",
                QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.PicturesLocation), #QtCore.QDir.currentPath(),
                "Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
            if filename is '':return    #用户按了取消.
            self.image_list_all_dir = []
            image_names=[]
            ls=filename.split('/')
            image_names.append(ls.pop())
            folder_name="/"
            folder_name= folder_name.join(ls)
            print("opening folder")
            print(folder_name)
            print("opening image")
            print(image_names)
            self.image_last_dir=[]
            self.folder=folder_name
            self.image_poi=0
            for image_name in image_names:
                self.image_list_all_dir.append(folder_name + '/' + image_name)
                self.image_last_dir.append(image_name)

        if filename:
            for poly in Allpoly.all_poly:
                print("Reset Item ")
                self.m_scene.removeItem(poly)
                for grip in poly.m_items:
                    self.m_scene.removeItem(grip)
            del Allpoly.all_poly[:]
            self.dock1_listwidget.clear()
            print("loading ")

            self.m_scene.file_name=filename
            ret_img=self.Multiband2Array(filename)
            if self.folder is None:self.folder = filename.split("\\")[0]

            print("read successfully step 1")
            if len(ret_img.shape)<3:
                lis=list(ret_img.shape)
                lis.append(1)
                ret_img.shape=tuple(lis)

            self.allBandImage=ret_img.copy()
            self.bandList=[]
            for i in range(ret_img.shape[2]):
                self.bandList.append(i)
            if ret_img.shape[2]>3:
                self.m_scene.cv_img=ret_img[:,:,0:3]
                self.m_scene.cv_nir_img=np.zeros((ret_img.shape[0],ret_img.shape[1],3),dtype=np.uint8)
                self.m_scene.cv_nir_img[:,:,0]=ret_img[:,:,3].copy()
                self.m_scene.cv_nir_img[:,:,1]=ret_img[:,:,0].copy()
                self.m_scene.cv_nir_img[:,:,2]=ret_img[:,:,1].copy()
            else:
                self.m_scene.cv_img=ret_img
                self.m_scene.cv_nir_img=None


            print("GDAL reading complete")
            print(ret_img.shape)

            self.m_scene.load_image(self.m_scene.cv_img)
            self.setWindowTitle(filename)
            self.load_item_from_txt()
            self.m_view.fitInView(self.m_scene.image_item, QtCore.Qt.KeepAspectRatio)
            self.m_view.centerOn(self.m_scene.image_item)
    def load_item_from_txt(self):
       
        img_name= self.image_last_dir[self.image_poi]
        folder_name=self.folder
        ls=img_name.split(".")
        ls.pop()
        mask_name="."
        mask_name=mask_name.join(ls)
        mask_name= mask_name+"_mask.txt"
        #print("load from txt ",folder_name,mask_name)
        if  not os.path.isfile(folder_name+"/mask/"+mask_name):return
        with open(folder_name+"/mask/"+mask_name,"r") as file:
            lines=file.readlines()
            for line in lines:
                line=line[:-1:]
                ls=line.split(" ")
                if ls[0]=="point":
                    ls=ls[1:]
                    str=" "
                    str=str.join(ls)
                    ls=str.split("$")
                    exterior=[]
                    for group in ls:
                        ls2=group.split(" ")
                        tmp=[]
                        for poi in ls2:
                            poi=poi[1:]
                            poi=poi[:-1:]
                            x_y_list=poi.split(",")
                            x=float(x_y_list[0])
                            y=float(x_y_list[1])
                            tmp.append((x,y))
                        exterior.append(tmp)

                    self.m_scene.setCurrentInstruction(Instructions.Polygon_Instruction)

                    self.m_scene.polygon_item.MySetFatherSonPolygon(exterior)
                else:
                    self.m_scene.polygon_item.my_color=int(ls[1])
                    self.m_scene.setCurrentInstruction(Instructions.Polygon_Finish)
    def stretch_n(self,bands, lower_percent=2, higher_percent=98):
        # print(bands.dtype)
        # 一定要使用float32类型，原因有两个：1、Keras不支持float64运算；2、float32运算要好于uint16
        out = np.zeros_like(bands).astype(np.float32)
        # print(out.dtype)
        for i in range(bands.shape[2]):
            # 这里直接拉伸到[0,1]之间，不需要先拉伸到[0,255]后面再转
            a = 0
            b = 1
            # 计算百分位数（从小到大排序之后第 percent% 的数）
            c = np.percentile(bands[:, :, i], lower_percent)
            d = np.percentile(bands[:, :, i], higher_percent)
            t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
            t[t < a] = a
            t[t > b] = b
            out[:, :, i] = t

        return out

    # 多波段图像(遥感图像)提取每个波段信息转换成数组（波段数>=4 或者 波段数<=3）
    # 一般的方法如：opencv，PIL，skimage 最多只能读取3个波段
    # path 图像的路径
    # return： 图像数组 注意 返回的假若是三通道图，那么返回的通道顺序是RGB，
    # 假若读到4通道图，我们规定原始的四通道tif必须存储为 RGB NIR， 所以返回的也是这个顺序
    # 若通道大于4，那么我们只返回4通道，舍弃剩余的通道
    # 若像素值不在0-255 范围内，本函数会进行线性拉伸到0-255.

    def Multiband2Array(self,path):
        print("using gdal")
        src_ds = gdal.Open(path)
        print("gdal suc")
        if src_ds is None:
            print('Unable to open %s'% path)
            sys.exit(1)

        xcount=src_ds.RasterXSize # 宽度
        ycount=src_ds.RasterYSize # 高度
        ibands=src_ds.RasterCount # 波段数

        # print "[ RASTER BAND COUNT ]: ", ibands
        for band in range(ibands):
            band += 1
            # print "[ GETTING BAND ]: ", band
            srcband = src_ds.GetRasterBand(band) # 获取该波段
            if srcband is None:
                continue

            # Read raster as arrays 类似RasterIO（C++）
            dataraster = srcband.ReadAsArray(0, 0, xcount, ycount).astype(np.float32) # 这里得到的矩阵大小为 ycount x xcount
            if band==1:
                data=dataraster.reshape((ycount,xcount,1))
            else:
                # 将每个波段的数组很并到一个3维数组中
                data=np.append(data,dataraster.reshape((ycount,xcount,1)),axis=2)
        if data.shape[2]>4:data=data[:,:,0:4]
        if np.max(data)>255:
            data=np.uint8(self.stretch_n(np.float32(data))*255)
            #cv2.imwrite("ts.tiff",data)

        return np.uint8(data)

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = AnnotationWindow()
    w.resize(1024, 768)
    w.show()
    sys.exit(app.exec_())
