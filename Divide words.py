import numpy as np
import cv2
import pyocr
import PIL.Image
from skimage.measure import label


class Image(np.ndarray):
	"""class for workig with whole images"""
	def __new__(subt, *args):
		return args[0].view(subt)

	def resize(self,to_y = 100.):
		ratio = to_y/img.shape[0]
		self = cv2.resize(self, None, fx=ratio, fy=ratio)

	def detect_edges(self,thresh_1=50,thresh_2=100):
		""""applying Canny edge detection to the image"""
		self.edges = cv2.Canny(self, thresh_1, thresh_2) / 255
	
	def parse(self):
		parsed = pyocr.tesseract.image_to_string(self.img(),lang="ukr",builder=pyocr.builders.WordBoxBuilder()) 
		if len(parsed) > 1:
			print "Multiple words found!"
			self.n_words = len(parsed)
			self.words = [self[x.position[0][1]:x.position[1][1],x.position[0][0]:x.position[1][0]] for x in parsed]
			for i in range(self.n_words):
				self.words[i].text = parsed[i].content
			self.text = ''.join([x.text for x in self.words])
			return len(self.words)
		else:
			if len(parsed) == 1:
				self.words = [self[parsed[0].position[0][1]:parsed[0].position[1][1],parsed[0].position[0][0]:parsed[0].position[1][0]]]
				self.words[0].text = parsed[0].content
				self.n_words = 1
				self.box = parsed[0].position
				self.text = parsed[0].content
				return 1
			else:
				print "Nothing found"
				return 0
	def area(self):
		return self.shape[0]*self.shape[1]
	def _repr_png_(self):
		return PIL.Image.fromarray(self)._repr_png_()

def get_elem(labeled, n_elem,image=a,return_pos = False):
    min0 = min(np.argwhere(labeled == n_elem), key=lambda x: x[0])[0]
    min1 = min(np.argwhere(labeled == n_elem), key=lambda x: x[1])[1]
    max0 = max(np.argwhere(labeled == n_elem), key=lambda x: x[0])[0]
    max1 = max(np.argwhere(labeled == n_elem), key=lambda x: x[1])[1]
    return (image[min0:max0,min1:max1], (min0,max0,min1,max1))

def im_resize(image, dest_y = 70):
    ratio  = float(dest_y) / image.shape[0]
    return Image(cv2.resize(image, None, fx = ratio, fy = ratio))

def to_square(image, sq_side = 70):
    sq = np.ones((sq_side,sq_side)) * 255
    margin =(sq_side - image.shape[1])/2
    sq[:,margin:margin+image.shape[1]] = image
    return Image(sq.astype(np.uint8))


cur_dir = getcwd()
# pdb.set_trace()
for path in argv[1:]:
	ptpdb.set_trace()
	img = Image(cv2.imread(cur_dir + '/' + path, 0))
	workdir = '/home/wolterlw/Desktop/bills/colored/letters' 

	info  = img.parse()
	b = (img < 10).astype(float) * 255
	b = cv2.dilate(b,np.ones([1,1]))
	labs, n_labs = label(b,background=0 ,connectivity=2, return_num=True)
	del b

	words = [x for x in img.words]

	letters = [get_elem(labs,i) for i in range(n_labs+1) if filter(a,get_elem(labs,i)[0])]
	letters = sorted(letters, key = lambda x: x[1][2])
	i = 0
	for letter in letters:
		letter = im_resize(letter[0])
		letter = to_square(letter)
		cv2.imwrite(workdir +'/letters_'+str(i)+'_'+path, img.get_t(row,word))
		i+=1