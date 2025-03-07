�<
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��3
x
dens_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedens_1/kernel
q
!dens_1/kernel/Read/ReadVariableOpReadVariableOpdens_1/kernel* 
_output_shapes
:
��*
dtype0
o
dens_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedens_1/bias
h
dens_1/bias/Read/ReadVariableOpReadVariableOpdens_1/bias*
_output_shapes	
:�*
dtype0
x
dens_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedens_2/kernel
q
!dens_2/kernel/Read/ReadVariableOpReadVariableOpdens_2/kernel* 
_output_shapes
:
��*
dtype0
o
dens_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedens_2/bias
h
dens_2/bias/Read/ReadVariableOpReadVariableOpdens_2/bias*
_output_shapes	
:�*
dtype0
x
dens_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedens_3/kernel
q
!dens_3/kernel/Read/ReadVariableOpReadVariableOpdens_3/kernel* 
_output_shapes
:
��*
dtype0
o
dens_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedens_3/bias
h
dens_3/bias/Read/ReadVariableOpReadVariableOpdens_3/bias*
_output_shapes	
:�*
dtype0
x
dens_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedens_4/kernel
q
!dens_4/kernel/Read/ReadVariableOpReadVariableOpdens_4/kernel* 
_output_shapes
:
��*
dtype0
o
dens_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedens_4/bias
h
dens_4/bias/Read/ReadVariableOpReadVariableOpdens_4/bias*
_output_shapes	
:�*
dtype0
x
dens_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedens_5/kernel
q
!dens_5/kernel/Read/ReadVariableOpReadVariableOpdens_5/kernel* 
_output_shapes
:
��*
dtype0
o
dens_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedens_5/bias
h
dens_5/bias/Read/ReadVariableOpReadVariableOpdens_5/bias*
_output_shapes	
:�*
dtype0
x
dens_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedens_6/kernel
q
!dens_6/kernel/Read/ReadVariableOpReadVariableOpdens_6/kernel* 
_output_shapes
:
��*
dtype0
o
dens_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedens_6/bias
h
dens_6/bias/Read/ReadVariableOpReadVariableOpdens_6/bias*
_output_shapes	
:�*
dtype0
x
dens_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedens_7/kernel
q
!dens_7/kernel/Read/ReadVariableOpReadVariableOpdens_7/kernel* 
_output_shapes
:
��*
dtype0
o
dens_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedens_7/bias
h
dens_7/bias/Read/ReadVariableOpReadVariableOpdens_7/bias*
_output_shapes	
:�*
dtype0
w
head_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namehead_1/kernel
p
!head_1/kernel/Read/ReadVariableOpReadVariableOphead_1/kernel*
_output_shapes
:	�*
dtype0
n
head_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehead_1/bias
g
head_1/bias/Read/ReadVariableOpReadVariableOphead_1/bias*
_output_shapes
:*
dtype0
w
head_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namehead_2/kernel
p
!head_2/kernel/Read/ReadVariableOpReadVariableOphead_2/kernel*
_output_shapes
:	�*
dtype0
n
head_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehead_2/bias
g
head_2/bias/Read/ReadVariableOpReadVariableOphead_2/bias*
_output_shapes
:*
dtype0
w
head_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namehead_3/kernel
p
!head_3/kernel/Read/ReadVariableOpReadVariableOphead_3/kernel*
_output_shapes
:	�*
dtype0
n
head_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehead_3/bias
g
head_3/bias/Read/ReadVariableOpReadVariableOphead_3/bias*
_output_shapes
:*
dtype0
w
head_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namehead_4/kernel
p
!head_4/kernel/Read/ReadVariableOpReadVariableOphead_4/kernel*
_output_shapes
:	�*
dtype0
n
head_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehead_4/bias
g
head_4/bias/Read/ReadVariableOpReadVariableOphead_4/bias*
_output_shapes
:*
dtype0
w
head_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namehead_5/kernel
p
!head_5/kernel/Read/ReadVariableOpReadVariableOphead_5/kernel*
_output_shapes
:	�*
dtype0
n
head_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehead_5/bias
g
head_5/bias/Read/ReadVariableOpReadVariableOphead_5/bias*
_output_shapes
:*
dtype0
w
head_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namehead_6/kernel
p
!head_6/kernel/Read/ReadVariableOpReadVariableOphead_6/kernel*
_output_shapes
:	�*
dtype0
n
head_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehead_6/bias
g
head_6/bias/Read/ReadVariableOpReadVariableOphead_6/bias*
_output_shapes
:*
dtype0
w
head_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namehead_7/kernel
p
!head_7/kernel/Read/ReadVariableOpReadVariableOphead_7/kernel*
_output_shapes
:	�*
dtype0
n
head_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehead_7/bias
g
head_7/bias/Read/ReadVariableOpReadVariableOphead_7/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
z
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_1/kernel
s
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*"
_output_shapes
: *
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
: *
dtype0
z
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv_2/kernel
s
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*"
_output_shapes
: @*
dtype0
n
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_2/bias
g
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes
:@*
dtype0
z
conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*
shared_nameconv_3/kernel
s
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*"
_output_shapes
:@`*
dtype0
n
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv_3/bias
g
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes
:`*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_9
[
total_9/Read/ReadVariableOpReadVariableOptotal_9*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
d
total_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_10
]
total_10/Read/ReadVariableOpReadVariableOptotal_10*
_output_shapes
: *
dtype0
d
count_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_10
]
count_10/Read/ReadVariableOpReadVariableOpcount_10*
_output_shapes
: *
dtype0
d
total_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_11
]
total_11/Read/ReadVariableOpReadVariableOptotal_11*
_output_shapes
: *
dtype0
d
count_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_11
]
count_11/Read/ReadVariableOpReadVariableOpcount_11*
_output_shapes
: *
dtype0
d
total_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_12
]
total_12/Read/ReadVariableOpReadVariableOptotal_12*
_output_shapes
: *
dtype0
d
count_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_12
]
count_12/Read/ReadVariableOpReadVariableOpcount_12*
_output_shapes
: *
dtype0
d
total_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_13
]
total_13/Read/ReadVariableOpReadVariableOptotal_13*
_output_shapes
: *
dtype0
d
count_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_13
]
count_13/Read/ReadVariableOpReadVariableOpcount_13*
_output_shapes
: *
dtype0
d
total_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_14
]
total_14/Read/ReadVariableOpReadVariableOptotal_14*
_output_shapes
: *
dtype0
d
count_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_14
]
count_14/Read/ReadVariableOpReadVariableOpcount_14*
_output_shapes
: *
dtype0
�
Adam/dens_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_1/kernel/m

(Adam/dens_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dens_1/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/dens_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_1/bias/m
v
&Adam/dens_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dens_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dens_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_2/kernel/m

(Adam/dens_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dens_2/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/dens_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_2/bias/m
v
&Adam/dens_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dens_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dens_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_3/kernel/m

(Adam/dens_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dens_3/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/dens_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_3/bias/m
v
&Adam/dens_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dens_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dens_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_4/kernel/m

(Adam/dens_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dens_4/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/dens_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_4/bias/m
v
&Adam/dens_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dens_4/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dens_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_5/kernel/m

(Adam/dens_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dens_5/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/dens_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_5/bias/m
v
&Adam/dens_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dens_5/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dens_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_6/kernel/m

(Adam/dens_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dens_6/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/dens_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_6/bias/m
v
&Adam/dens_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dens_6/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dens_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_7/kernel/m

(Adam/dens_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dens_7/kernel/m* 
_output_shapes
:
��*
dtype0
}
Adam/dens_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_7/bias/m
v
&Adam/dens_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dens_7/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/head_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_1/kernel/m
~
(Adam/head_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/head_1/kernel/m*
_output_shapes
:	�*
dtype0
|
Adam/head_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_1/bias/m
u
&Adam/head_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/head_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/head_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_2/kernel/m
~
(Adam/head_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/head_2/kernel/m*
_output_shapes
:	�*
dtype0
|
Adam/head_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_2/bias/m
u
&Adam/head_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/head_2/bias/m*
_output_shapes
:*
dtype0
�
Adam/head_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_3/kernel/m
~
(Adam/head_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/head_3/kernel/m*
_output_shapes
:	�*
dtype0
|
Adam/head_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_3/bias/m
u
&Adam/head_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/head_3/bias/m*
_output_shapes
:*
dtype0
�
Adam/head_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_4/kernel/m
~
(Adam/head_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/head_4/kernel/m*
_output_shapes
:	�*
dtype0
|
Adam/head_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_4/bias/m
u
&Adam/head_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/head_4/bias/m*
_output_shapes
:*
dtype0
�
Adam/head_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_5/kernel/m
~
(Adam/head_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/head_5/kernel/m*
_output_shapes
:	�*
dtype0
|
Adam/head_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_5/bias/m
u
&Adam/head_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/head_5/bias/m*
_output_shapes
:*
dtype0
�
Adam/head_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_6/kernel/m
~
(Adam/head_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/head_6/kernel/m*
_output_shapes
:	�*
dtype0
|
Adam/head_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_6/bias/m
u
&Adam/head_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/head_6/bias/m*
_output_shapes
:*
dtype0
�
Adam/head_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_7/kernel/m
~
(Adam/head_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/head_7/kernel/m*
_output_shapes
:	�*
dtype0
|
Adam/head_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_7/bias/m
u
&Adam/head_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/head_7/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv_1/kernel/m
�
(Adam/conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_1/kernel/m*"
_output_shapes
: *
dtype0
|
Adam/conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv_1/bias/m
u
&Adam/conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_1/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*%
shared_nameAdam/conv_2/kernel/m
�
(Adam/conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_2/kernel/m*"
_output_shapes
: @*
dtype0
|
Adam/conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_2/bias/m
u
&Adam/conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_2/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*%
shared_nameAdam/conv_3/kernel/m
�
(Adam/conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_3/kernel/m*"
_output_shapes
:@`*
dtype0
|
Adam/conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*#
shared_nameAdam/conv_3/bias/m
u
&Adam/conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_3/bias/m*
_output_shapes
:`*
dtype0
�
Adam/dens_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_1/kernel/v

(Adam/dens_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dens_1/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/dens_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_1/bias/v
v
&Adam/dens_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dens_1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dens_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_2/kernel/v

(Adam/dens_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dens_2/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/dens_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_2/bias/v
v
&Adam/dens_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dens_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dens_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_3/kernel/v

(Adam/dens_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dens_3/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/dens_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_3/bias/v
v
&Adam/dens_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dens_3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dens_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_4/kernel/v

(Adam/dens_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dens_4/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/dens_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_4/bias/v
v
&Adam/dens_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dens_4/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dens_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_5/kernel/v

(Adam/dens_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dens_5/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/dens_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_5/bias/v
v
&Adam/dens_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dens_5/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dens_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_6/kernel/v

(Adam/dens_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dens_6/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/dens_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_6/bias/v
v
&Adam/dens_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dens_6/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dens_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_nameAdam/dens_7/kernel/v

(Adam/dens_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dens_7/kernel/v* 
_output_shapes
:
��*
dtype0
}
Adam/dens_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_nameAdam/dens_7/bias/v
v
&Adam/dens_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dens_7/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/head_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_1/kernel/v
~
(Adam/head_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/head_1/kernel/v*
_output_shapes
:	�*
dtype0
|
Adam/head_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_1/bias/v
u
&Adam/head_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/head_1/bias/v*
_output_shapes
:*
dtype0
�
Adam/head_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_2/kernel/v
~
(Adam/head_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/head_2/kernel/v*
_output_shapes
:	�*
dtype0
|
Adam/head_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_2/bias/v
u
&Adam/head_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/head_2/bias/v*
_output_shapes
:*
dtype0
�
Adam/head_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_3/kernel/v
~
(Adam/head_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/head_3/kernel/v*
_output_shapes
:	�*
dtype0
|
Adam/head_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_3/bias/v
u
&Adam/head_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/head_3/bias/v*
_output_shapes
:*
dtype0
�
Adam/head_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_4/kernel/v
~
(Adam/head_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/head_4/kernel/v*
_output_shapes
:	�*
dtype0
|
Adam/head_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_4/bias/v
u
&Adam/head_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/head_4/bias/v*
_output_shapes
:*
dtype0
�
Adam/head_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_5/kernel/v
~
(Adam/head_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/head_5/kernel/v*
_output_shapes
:	�*
dtype0
|
Adam/head_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_5/bias/v
u
&Adam/head_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/head_5/bias/v*
_output_shapes
:*
dtype0
�
Adam/head_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_6/kernel/v
~
(Adam/head_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/head_6/kernel/v*
_output_shapes
:	�*
dtype0
|
Adam/head_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_6/bias/v
u
&Adam/head_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/head_6/bias/v*
_output_shapes
:*
dtype0
�
Adam/head_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_nameAdam/head_7/kernel/v
~
(Adam/head_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/head_7/kernel/v*
_output_shapes
:	�*
dtype0
|
Adam/head_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/head_7/bias/v
u
&Adam/head_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/head_7/bias/v*
_output_shapes
:*
dtype0
�
Adam/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv_1/kernel/v
�
(Adam/conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_1/kernel/v*"
_output_shapes
: *
dtype0
|
Adam/conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv_1/bias/v
u
&Adam/conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_1/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*%
shared_nameAdam/conv_2/kernel/v
�
(Adam/conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_2/kernel/v*"
_output_shapes
: @*
dtype0
|
Adam/conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv_2/bias/v
u
&Adam/conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_2/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@`*%
shared_nameAdam/conv_3/kernel/v
�
(Adam/conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_3/kernel/v*"
_output_shapes
:@`*
dtype0
|
Adam/conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*#
shared_nameAdam/conv_3/bias/v
u
&Adam/conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_3/bias/v*
_output_shapes
:`*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer-8

layer_with_weights-1

layer-9
layer_with_weights-2
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
 
 
 
 
 
�
layer-0
 layer_with_weights-0
 layer-1
!layer-2
"layer_with_weights-1
"layer-3
#layer-4
$layer_with_weights-2
$layer-5
%layer-6
&	variables
'trainable_variables
(regularization_losses
)	keras_api
y
*layer-0
+layer-1
,layer-2
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
h

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
h

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
h

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
h

Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
h

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
h

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
h

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
h

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
h

ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
m

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate1m�2m�7m�8m�=m�>m�Cm�Dm�Im�Jm�Om�Pm�Um�Vm�[m�\m�am�bm�gm�hm�mm�nm�sm�tm�ym�zm�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�1v�2v�7v�8v�=v�>v�Cv�Dv�Iv�Jv�Ov�Pv�Uv�Vv�[v�\v�av�bv�gv�hv�mv�nv�sv�tv�yv�zv�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�
 
�
�0
�1
�2
�3
�4
�5
16
27
78
89
=10
>11
C12
D13
I14
J15
O16
P17
U18
V19
[20
\21
a22
b23
g24
h25
m26
n27
s28
t29
y30
z31
32
�33
�
�0
�1
�2
�3
�4
�5
16
27
78
89
=10
>11
C12
D13
I14
J15
O16
P17
U18
V19
[20
\21
a22
b23
g24
h25
m26
n27
s28
t29
y30
z31
32
�33
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
	variables
trainable_variables
�layers
regularization_losses
 
 
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
0
�0
�1
�2
�3
�4
�5
0
�0
�1
�2
�3
�4
�5
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
&	variables
'trainable_variables
�layers
(regularization_losses
 
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
 
 
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
-	variables
.trainable_variables
�layers
/regularization_losses
YW
VARIABLE_VALUEdens_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdens_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
3	variables
4trainable_variables
�layers
5regularization_losses
YW
VARIABLE_VALUEdens_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdens_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
9	variables
:trainable_variables
�layers
;regularization_losses
YW
VARIABLE_VALUEdens_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdens_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
?	variables
@trainable_variables
�layers
Aregularization_losses
YW
VARIABLE_VALUEdens_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdens_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
E	variables
Ftrainable_variables
�layers
Gregularization_losses
YW
VARIABLE_VALUEdens_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdens_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1

I0
J1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
K	variables
Ltrainable_variables
�layers
Mregularization_losses
YW
VARIABLE_VALUEdens_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdens_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

O0
P1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
Q	variables
Rtrainable_variables
�layers
Sregularization_losses
YW
VARIABLE_VALUEdens_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdens_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1

U0
V1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
W	variables
Xtrainable_variables
�layers
Yregularization_losses
YW
VARIABLE_VALUEhead_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEhead_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

[0
\1

[0
\1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
]	variables
^trainable_variables
�layers
_regularization_losses
YW
VARIABLE_VALUEhead_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEhead_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

a0
b1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
c	variables
dtrainable_variables
�layers
eregularization_losses
ZX
VARIABLE_VALUEhead_3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhead_3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
i	variables
jtrainable_variables
�layers
kregularization_losses
ZX
VARIABLE_VALUEhead_4/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhead_4/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

m0
n1

m0
n1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
o	variables
ptrainable_variables
�layers
qregularization_losses
ZX
VARIABLE_VALUEhead_5/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhead_5/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

s0
t1

s0
t1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
u	variables
vtrainable_variables
�layers
wregularization_losses
ZX
VARIABLE_VALUEhead_6/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhead_6/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

y0
z1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
{	variables
|trainable_variables
�layers
}regularization_losses
ZX
VARIABLE_VALUEhead_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEhead_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

0
�1

0
�1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv_1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv_2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv_2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv_3/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv_3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
}
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22

�0
�1

�0
�1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
 
 
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses

�0
�1

�0
�1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
 
 
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses

�0
�1

�0
�1
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
 
 
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
 
 
 
 
1
0
 1
!2
"3
#4
$5
%6
 
 
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
 
 
 
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
 
 
 
 

*0
+1
,2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_74keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_84keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_94keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
SQ
VARIABLE_VALUEtotal_105keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_105keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
SQ
VARIABLE_VALUEtotal_115keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_115keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
SQ
VARIABLE_VALUEtotal_125keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_125keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
SQ
VARIABLE_VALUEtotal_135keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_135keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
SQ
VARIABLE_VALUEtotal_145keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_145keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
|z
VARIABLE_VALUEAdam/dens_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_6/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_6/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_7/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_7/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/head_1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/head_1/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/head_2/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/head_2/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_3/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_3/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_4/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_4/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_5/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_5/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_6/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_6/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_7/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_7/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv_1/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv_2/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv_2/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv_3/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv_3/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_6/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_6/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dens_7/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dens_7/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/head_1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/head_1/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/head_2/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/head_2/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_3/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_3/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_4/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_4/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_5/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_5/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_6/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_6/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/head_7/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/head_7/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv_1/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv_2/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv_2/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv_3/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv_3/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
serving_default_input_2Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
serving_default_input_3Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
serving_default_input_4Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
serving_default_input_5Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
serving_default_input_6Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
serving_default_input_7Placeholder*,
_output_shapes
:����������*
dtype0*!
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3serving_default_input_4serving_default_input_5serving_default_input_6serving_default_input_7conv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasdens_7/kerneldens_7/biasdens_6/kerneldens_6/biasdens_5/kerneldens_5/biasdens_4/kerneldens_4/biasdens_3/kerneldens_3/biasdens_2/kerneldens_2/biasdens_1/kerneldens_1/biashead_7/kernelhead_7/biashead_6/kernelhead_6/biashead_5/kernelhead_5/biashead_4/kernelhead_4/biashead_3/kernelhead_3/biashead_2/kernelhead_2/biashead_1/kernelhead_1/bias*4
Tin-
+2)*
Tout
	2*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������*D
_read_only_resource_inputs&
$"	
 !"#$%&'(*2
config_proto" 

CPU

GPU2*0,1J 8� *-
f(R&
$__inference_signature_wrapper_218350
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�*
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!dens_1/kernel/Read/ReadVariableOpdens_1/bias/Read/ReadVariableOp!dens_2/kernel/Read/ReadVariableOpdens_2/bias/Read/ReadVariableOp!dens_3/kernel/Read/ReadVariableOpdens_3/bias/Read/ReadVariableOp!dens_4/kernel/Read/ReadVariableOpdens_4/bias/Read/ReadVariableOp!dens_5/kernel/Read/ReadVariableOpdens_5/bias/Read/ReadVariableOp!dens_6/kernel/Read/ReadVariableOpdens_6/bias/Read/ReadVariableOp!dens_7/kernel/Read/ReadVariableOpdens_7/bias/Read/ReadVariableOp!head_1/kernel/Read/ReadVariableOphead_1/bias/Read/ReadVariableOp!head_2/kernel/Read/ReadVariableOphead_2/bias/Read/ReadVariableOp!head_3/kernel/Read/ReadVariableOphead_3/bias/Read/ReadVariableOp!head_4/kernel/Read/ReadVariableOphead_4/bias/Read/ReadVariableOp!head_5/kernel/Read/ReadVariableOphead_5/bias/Read/ReadVariableOp!head_6/kernel/Read/ReadVariableOphead_6/bias/Read/ReadVariableOp!head_7/kernel/Read/ReadVariableOphead_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_7/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_8/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_9/Read/ReadVariableOpcount_9/Read/ReadVariableOptotal_10/Read/ReadVariableOpcount_10/Read/ReadVariableOptotal_11/Read/ReadVariableOpcount_11/Read/ReadVariableOptotal_12/Read/ReadVariableOpcount_12/Read/ReadVariableOptotal_13/Read/ReadVariableOpcount_13/Read/ReadVariableOptotal_14/Read/ReadVariableOpcount_14/Read/ReadVariableOp(Adam/dens_1/kernel/m/Read/ReadVariableOp&Adam/dens_1/bias/m/Read/ReadVariableOp(Adam/dens_2/kernel/m/Read/ReadVariableOp&Adam/dens_2/bias/m/Read/ReadVariableOp(Adam/dens_3/kernel/m/Read/ReadVariableOp&Adam/dens_3/bias/m/Read/ReadVariableOp(Adam/dens_4/kernel/m/Read/ReadVariableOp&Adam/dens_4/bias/m/Read/ReadVariableOp(Adam/dens_5/kernel/m/Read/ReadVariableOp&Adam/dens_5/bias/m/Read/ReadVariableOp(Adam/dens_6/kernel/m/Read/ReadVariableOp&Adam/dens_6/bias/m/Read/ReadVariableOp(Adam/dens_7/kernel/m/Read/ReadVariableOp&Adam/dens_7/bias/m/Read/ReadVariableOp(Adam/head_1/kernel/m/Read/ReadVariableOp&Adam/head_1/bias/m/Read/ReadVariableOp(Adam/head_2/kernel/m/Read/ReadVariableOp&Adam/head_2/bias/m/Read/ReadVariableOp(Adam/head_3/kernel/m/Read/ReadVariableOp&Adam/head_3/bias/m/Read/ReadVariableOp(Adam/head_4/kernel/m/Read/ReadVariableOp&Adam/head_4/bias/m/Read/ReadVariableOp(Adam/head_5/kernel/m/Read/ReadVariableOp&Adam/head_5/bias/m/Read/ReadVariableOp(Adam/head_6/kernel/m/Read/ReadVariableOp&Adam/head_6/bias/m/Read/ReadVariableOp(Adam/head_7/kernel/m/Read/ReadVariableOp&Adam/head_7/bias/m/Read/ReadVariableOp(Adam/conv_1/kernel/m/Read/ReadVariableOp&Adam/conv_1/bias/m/Read/ReadVariableOp(Adam/conv_2/kernel/m/Read/ReadVariableOp&Adam/conv_2/bias/m/Read/ReadVariableOp(Adam/conv_3/kernel/m/Read/ReadVariableOp&Adam/conv_3/bias/m/Read/ReadVariableOp(Adam/dens_1/kernel/v/Read/ReadVariableOp&Adam/dens_1/bias/v/Read/ReadVariableOp(Adam/dens_2/kernel/v/Read/ReadVariableOp&Adam/dens_2/bias/v/Read/ReadVariableOp(Adam/dens_3/kernel/v/Read/ReadVariableOp&Adam/dens_3/bias/v/Read/ReadVariableOp(Adam/dens_4/kernel/v/Read/ReadVariableOp&Adam/dens_4/bias/v/Read/ReadVariableOp(Adam/dens_5/kernel/v/Read/ReadVariableOp&Adam/dens_5/bias/v/Read/ReadVariableOp(Adam/dens_6/kernel/v/Read/ReadVariableOp&Adam/dens_6/bias/v/Read/ReadVariableOp(Adam/dens_7/kernel/v/Read/ReadVariableOp&Adam/dens_7/bias/v/Read/ReadVariableOp(Adam/head_1/kernel/v/Read/ReadVariableOp&Adam/head_1/bias/v/Read/ReadVariableOp(Adam/head_2/kernel/v/Read/ReadVariableOp&Adam/head_2/bias/v/Read/ReadVariableOp(Adam/head_3/kernel/v/Read/ReadVariableOp&Adam/head_3/bias/v/Read/ReadVariableOp(Adam/head_4/kernel/v/Read/ReadVariableOp&Adam/head_4/bias/v/Read/ReadVariableOp(Adam/head_5/kernel/v/Read/ReadVariableOp&Adam/head_5/bias/v/Read/ReadVariableOp(Adam/head_6/kernel/v/Read/ReadVariableOp&Adam/head_6/bias/v/Read/ReadVariableOp(Adam/head_7/kernel/v/Read/ReadVariableOp&Adam/head_7/bias/v/Read/ReadVariableOp(Adam/conv_1/kernel/v/Read/ReadVariableOp&Adam/conv_1/bias/v/Read/ReadVariableOp(Adam/conv_2/kernel/v/Read/ReadVariableOp&Adam/conv_2/bias/v/Read/ReadVariableOp(Adam/conv_3/kernel/v/Read/ReadVariableOp&Adam/conv_3/bias/v/Read/ReadVariableOpConst*�
Tin�
�2�	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *(
f#R!
__inference__traced_save_220688
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedens_1/kerneldens_1/biasdens_2/kerneldens_2/biasdens_3/kerneldens_3/biasdens_4/kerneldens_4/biasdens_5/kerneldens_5/biasdens_6/kerneldens_6/biasdens_7/kerneldens_7/biashead_1/kernelhead_1/biashead_2/kernelhead_2/biashead_3/kernelhead_3/biashead_4/kernelhead_4/biashead_5/kernelhead_5/biashead_6/kernelhead_6/biashead_7/kernelhead_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biastotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4total_5count_5total_6count_6total_7count_7total_8count_8total_9count_9total_10count_10total_11count_11total_12count_12total_13count_13total_14count_14Adam/dens_1/kernel/mAdam/dens_1/bias/mAdam/dens_2/kernel/mAdam/dens_2/bias/mAdam/dens_3/kernel/mAdam/dens_3/bias/mAdam/dens_4/kernel/mAdam/dens_4/bias/mAdam/dens_5/kernel/mAdam/dens_5/bias/mAdam/dens_6/kernel/mAdam/dens_6/bias/mAdam/dens_7/kernel/mAdam/dens_7/bias/mAdam/head_1/kernel/mAdam/head_1/bias/mAdam/head_2/kernel/mAdam/head_2/bias/mAdam/head_3/kernel/mAdam/head_3/bias/mAdam/head_4/kernel/mAdam/head_4/bias/mAdam/head_5/kernel/mAdam/head_5/bias/mAdam/head_6/kernel/mAdam/head_6/bias/mAdam/head_7/kernel/mAdam/head_7/bias/mAdam/conv_1/kernel/mAdam/conv_1/bias/mAdam/conv_2/kernel/mAdam/conv_2/bias/mAdam/conv_3/kernel/mAdam/conv_3/bias/mAdam/dens_1/kernel/vAdam/dens_1/bias/vAdam/dens_2/kernel/vAdam/dens_2/bias/vAdam/dens_3/kernel/vAdam/dens_3/bias/vAdam/dens_4/kernel/vAdam/dens_4/bias/vAdam/dens_5/kernel/vAdam/dens_5/bias/vAdam/dens_6/kernel/vAdam/dens_6/bias/vAdam/dens_7/kernel/vAdam/dens_7/bias/vAdam/head_1/kernel/vAdam/head_1/bias/vAdam/head_2/kernel/vAdam/head_2/bias/vAdam/head_3/kernel/vAdam/head_3/bias/vAdam/head_4/kernel/vAdam/head_4/bias/vAdam/head_5/kernel/vAdam/head_5/bias/vAdam/head_6/kernel/vAdam/head_6/bias/vAdam/head_7/kernel/vAdam/head_7/bias/vAdam/conv_1/kernel/vAdam/conv_1/bias/vAdam/conv_2/kernel/vAdam/conv_2/bias/vAdam/conv_3/kernel/vAdam/conv_3/bias/v*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *+
f&R$
"__inference__traced_restore_221109��/
�
�
'__inference_dens_2_layer_call_fn_219755

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_2_layer_call_and_return_conditional_losses_2170722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dens_6_layer_call_and_return_conditional_losses_217004

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_218067
input_1
input_2
input_3
input_4
input_5
input_6
input_7#
trunk__217910: 
trunk__217912: #
trunk__217914: @
trunk__217916:@#
trunk__217918:@`
trunk__217920:`!
dens_7_217972:
��
dens_7_217974:	�!
dens_6_217977:
��
dens_6_217979:	�!
dens_5_217982:
��
dens_5_217984:	�!
dens_4_217987:
��
dens_4_217989:	�!
dens_3_217992:
��
dens_3_217994:	�!
dens_2_217997:
��
dens_2_217999:	�!
dens_1_218002:
��
dens_1_218004:	� 
head_7_218007:	�
head_7_218009: 
head_6_218012:	�
head_6_218014: 
head_5_218017:	�
head_5_218019: 
head_4_218022:	�
head_4_218024: 
head_3_218027:	�
head_3_218029: 
head_2_218032:	�
head_2_218034: 
head_1_218037:	�
head_1_218039:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��/conv_1/kernel/Regularizer/Square/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOp�dens_1/StatefulPartitionedCall�dens_2/StatefulPartitionedCall�dens_3/StatefulPartitionedCall�dens_4/StatefulPartitionedCall�dens_5/StatefulPartitionedCall�dens_6/StatefulPartitionedCall�dens_7/StatefulPartitionedCall�head_1/StatefulPartitionedCall�head_2/StatefulPartitionedCall�head_3/StatefulPartitionedCall�head_4/StatefulPartitionedCall�head_5/StatefulPartitionedCall�head_6/StatefulPartitionedCall�head_7/StatefulPartitionedCall�trunk_/StatefulPartitionedCall� trunk_/StatefulPartitionedCall_1� trunk_/StatefulPartitionedCall_2� trunk_/StatefulPartitionedCall_3� trunk_/StatefulPartitionedCall_4� trunk_/StatefulPartitionedCall_5� trunk_/StatefulPartitionedCall_6�
trunk_/StatefulPartitionedCallStatefulPartitionedCallinput_7trunk__217910trunk__217912trunk__217914trunk__217916trunk__217918trunk__217920*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362 
trunk_/StatefulPartitionedCall�
 trunk_/StatefulPartitionedCall_1StatefulPartitionedCallinput_6trunk__217910trunk__217912trunk__217914trunk__217916trunk__217918trunk__217920*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_1�
 trunk_/StatefulPartitionedCall_2StatefulPartitionedCallinput_5trunk__217910trunk__217912trunk__217914trunk__217916trunk__217918trunk__217920*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_2�
 trunk_/StatefulPartitionedCall_3StatefulPartitionedCallinput_4trunk__217910trunk__217912trunk__217914trunk__217916trunk__217918trunk__217920*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_3�
 trunk_/StatefulPartitionedCall_4StatefulPartitionedCallinput_3trunk__217910trunk__217912trunk__217914trunk__217916trunk__217918trunk__217920*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_4�
 trunk_/StatefulPartitionedCall_5StatefulPartitionedCallinput_2trunk__217910trunk__217912trunk__217914trunk__217916trunk__217918trunk__217920*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_5�
 trunk_/StatefulPartitionedCall_6StatefulPartitionedCallinput_1trunk__217910trunk__217912trunk__217914trunk__217916trunk__217918trunk__217920*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_6�
 global_max_pool_/PartitionedCallPartitionedCall'trunk_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532"
 global_max_pool_/PartitionedCall�
"global_max_pool_/PartitionedCall_1PartitionedCall)trunk_/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_1�
"global_max_pool_/PartitionedCall_2PartitionedCall)trunk_/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_2�
"global_max_pool_/PartitionedCall_3PartitionedCall)trunk_/StatefulPartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_3�
"global_max_pool_/PartitionedCall_4PartitionedCall)trunk_/StatefulPartitionedCall_4:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_4�
"global_max_pool_/PartitionedCall_5PartitionedCall)trunk_/StatefulPartitionedCall_5:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_5�
"global_max_pool_/PartitionedCall_6PartitionedCall)trunk_/StatefulPartitionedCall_6:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_6�
dens_7/StatefulPartitionedCallStatefulPartitionedCall)global_max_pool_/PartitionedCall:output:0dens_7_217972dens_7_217974*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_7_layer_call_and_return_conditional_losses_2169872 
dens_7/StatefulPartitionedCall�
dens_6/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_1:output:0dens_6_217977dens_6_217979*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_6_layer_call_and_return_conditional_losses_2170042 
dens_6/StatefulPartitionedCall�
dens_5/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_2:output:0dens_5_217982dens_5_217984*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_5_layer_call_and_return_conditional_losses_2170212 
dens_5/StatefulPartitionedCall�
dens_4/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_3:output:0dens_4_217987dens_4_217989*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_4_layer_call_and_return_conditional_losses_2170382 
dens_4/StatefulPartitionedCall�
dens_3/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_4:output:0dens_3_217992dens_3_217994*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_3_layer_call_and_return_conditional_losses_2170552 
dens_3/StatefulPartitionedCall�
dens_2/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_5:output:0dens_2_217997dens_2_217999*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_2_layer_call_and_return_conditional_losses_2170722 
dens_2/StatefulPartitionedCall�
dens_1/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_6:output:0dens_1_218002dens_1_218004*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_1_layer_call_and_return_conditional_losses_2170892 
dens_1/StatefulPartitionedCall�
head_7/StatefulPartitionedCallStatefulPartitionedCall'dens_7/StatefulPartitionedCall:output:0head_7_218007head_7_218009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_7_layer_call_and_return_conditional_losses_2171062 
head_7/StatefulPartitionedCall�
head_6/StatefulPartitionedCallStatefulPartitionedCall'dens_6/StatefulPartitionedCall:output:0head_6_218012head_6_218014*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_6_layer_call_and_return_conditional_losses_2171232 
head_6/StatefulPartitionedCall�
head_5/StatefulPartitionedCallStatefulPartitionedCall'dens_5/StatefulPartitionedCall:output:0head_5_218017head_5_218019*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_5_layer_call_and_return_conditional_losses_2171402 
head_5/StatefulPartitionedCall�
head_4/StatefulPartitionedCallStatefulPartitionedCall'dens_4/StatefulPartitionedCall:output:0head_4_218022head_4_218024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_4_layer_call_and_return_conditional_losses_2171572 
head_4/StatefulPartitionedCall�
head_3/StatefulPartitionedCallStatefulPartitionedCall'dens_3/StatefulPartitionedCall:output:0head_3_218027head_3_218029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_3_layer_call_and_return_conditional_losses_2171742 
head_3/StatefulPartitionedCall�
head_2/StatefulPartitionedCallStatefulPartitionedCall'dens_2/StatefulPartitionedCall:output:0head_2_218032head_2_218034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_2_layer_call_and_return_conditional_losses_2171912 
head_2/StatefulPartitionedCall�
head_1/StatefulPartitionedCallStatefulPartitionedCall'dens_1/StatefulPartitionedCall:output:0head_1_218037head_1_218039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_1_layer_call_and_return_conditional_losses_2172082 
head_1/StatefulPartitionedCall�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__217910*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__217914*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__217918*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentity'head_1/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity'head_2/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity'head_3/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identity'head_4/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identity'head_5/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity'head_6/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identity'head_7/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp2@
dens_1/StatefulPartitionedCalldens_1/StatefulPartitionedCall2@
dens_2/StatefulPartitionedCalldens_2/StatefulPartitionedCall2@
dens_3/StatefulPartitionedCalldens_3/StatefulPartitionedCall2@
dens_4/StatefulPartitionedCalldens_4/StatefulPartitionedCall2@
dens_5/StatefulPartitionedCalldens_5/StatefulPartitionedCall2@
dens_6/StatefulPartitionedCalldens_6/StatefulPartitionedCall2@
dens_7/StatefulPartitionedCalldens_7/StatefulPartitionedCall2@
head_1/StatefulPartitionedCallhead_1/StatefulPartitionedCall2@
head_2/StatefulPartitionedCallhead_2/StatefulPartitionedCall2@
head_3/StatefulPartitionedCallhead_3/StatefulPartitionedCall2@
head_4/StatefulPartitionedCallhead_4/StatefulPartitionedCall2@
head_5/StatefulPartitionedCallhead_5/StatefulPartitionedCall2@
head_6/StatefulPartitionedCallhead_6/StatefulPartitionedCall2@
head_7/StatefulPartitionedCallhead_7/StatefulPartitionedCall2@
trunk_/StatefulPartitionedCalltrunk_/StatefulPartitionedCall2D
 trunk_/StatefulPartitionedCall_1 trunk_/StatefulPartitionedCall_12D
 trunk_/StatefulPartitionedCall_2 trunk_/StatefulPartitionedCall_22D
 trunk_/StatefulPartitionedCall_3 trunk_/StatefulPartitionedCall_32D
 trunk_/StatefulPartitionedCall_4 trunk_/StatefulPartitionedCall_42D
 trunk_/StatefulPartitionedCall_5 trunk_/StatefulPartitionedCall_52D
 trunk_/StatefulPartitionedCall_6 trunk_/StatefulPartitionedCall_6:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_2:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_3:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_4:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_5:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_6:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_7
�

�
B__inference_head_1_layer_call_and_return_conditional_losses_217208

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_217727

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6#
trunk__217570: 
trunk__217572: #
trunk__217574: @
trunk__217576:@#
trunk__217578:@`
trunk__217580:`!
dens_7_217632:
��
dens_7_217634:	�!
dens_6_217637:
��
dens_6_217639:	�!
dens_5_217642:
��
dens_5_217644:	�!
dens_4_217647:
��
dens_4_217649:	�!
dens_3_217652:
��
dens_3_217654:	�!
dens_2_217657:
��
dens_2_217659:	�!
dens_1_217662:
��
dens_1_217664:	� 
head_7_217667:	�
head_7_217669: 
head_6_217672:	�
head_6_217674: 
head_5_217677:	�
head_5_217679: 
head_4_217682:	�
head_4_217684: 
head_3_217687:	�
head_3_217689: 
head_2_217692:	�
head_2_217694: 
head_1_217697:	�
head_1_217699:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��/conv_1/kernel/Regularizer/Square/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOp�dens_1/StatefulPartitionedCall�dens_2/StatefulPartitionedCall�dens_3/StatefulPartitionedCall�dens_4/StatefulPartitionedCall�dens_5/StatefulPartitionedCall�dens_6/StatefulPartitionedCall�dens_7/StatefulPartitionedCall�head_1/StatefulPartitionedCall�head_2/StatefulPartitionedCall�head_3/StatefulPartitionedCall�head_4/StatefulPartitionedCall�head_5/StatefulPartitionedCall�head_6/StatefulPartitionedCall�head_7/StatefulPartitionedCall�trunk_/StatefulPartitionedCall� trunk_/StatefulPartitionedCall_1� trunk_/StatefulPartitionedCall_2� trunk_/StatefulPartitionedCall_3� trunk_/StatefulPartitionedCall_4� trunk_/StatefulPartitionedCall_5� trunk_/StatefulPartitionedCall_6�
trunk_/StatefulPartitionedCallStatefulPartitionedCallinputs_6trunk__217570trunk__217572trunk__217574trunk__217576trunk__217578trunk__217580*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092 
trunk_/StatefulPartitionedCall�
 trunk_/StatefulPartitionedCall_1StatefulPartitionedCallinputs_5trunk__217570trunk__217572trunk__217574trunk__217576trunk__217578trunk__217580*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_1�
 trunk_/StatefulPartitionedCall_2StatefulPartitionedCallinputs_4trunk__217570trunk__217572trunk__217574trunk__217576trunk__217578trunk__217580*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_2�
 trunk_/StatefulPartitionedCall_3StatefulPartitionedCallinputs_3trunk__217570trunk__217572trunk__217574trunk__217576trunk__217578trunk__217580*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_3�
 trunk_/StatefulPartitionedCall_4StatefulPartitionedCallinputs_2trunk__217570trunk__217572trunk__217574trunk__217576trunk__217578trunk__217580*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_4�
 trunk_/StatefulPartitionedCall_5StatefulPartitionedCallinputs_1trunk__217570trunk__217572trunk__217574trunk__217576trunk__217578trunk__217580*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_5�
 trunk_/StatefulPartitionedCall_6StatefulPartitionedCallinputstrunk__217570trunk__217572trunk__217574trunk__217576trunk__217578trunk__217580*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_6�
 global_max_pool_/PartitionedCallPartitionedCall'trunk_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752"
 global_max_pool_/PartitionedCall�
"global_max_pool_/PartitionedCall_1PartitionedCall)trunk_/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_1�
"global_max_pool_/PartitionedCall_2PartitionedCall)trunk_/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_2�
"global_max_pool_/PartitionedCall_3PartitionedCall)trunk_/StatefulPartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_3�
"global_max_pool_/PartitionedCall_4PartitionedCall)trunk_/StatefulPartitionedCall_4:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_4�
"global_max_pool_/PartitionedCall_5PartitionedCall)trunk_/StatefulPartitionedCall_5:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_5�
"global_max_pool_/PartitionedCall_6PartitionedCall)trunk_/StatefulPartitionedCall_6:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_6�
dens_7/StatefulPartitionedCallStatefulPartitionedCall)global_max_pool_/PartitionedCall:output:0dens_7_217632dens_7_217634*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_7_layer_call_and_return_conditional_losses_2169872 
dens_7/StatefulPartitionedCall�
dens_6/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_1:output:0dens_6_217637dens_6_217639*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_6_layer_call_and_return_conditional_losses_2170042 
dens_6/StatefulPartitionedCall�
dens_5/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_2:output:0dens_5_217642dens_5_217644*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_5_layer_call_and_return_conditional_losses_2170212 
dens_5/StatefulPartitionedCall�
dens_4/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_3:output:0dens_4_217647dens_4_217649*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_4_layer_call_and_return_conditional_losses_2170382 
dens_4/StatefulPartitionedCall�
dens_3/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_4:output:0dens_3_217652dens_3_217654*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_3_layer_call_and_return_conditional_losses_2170552 
dens_3/StatefulPartitionedCall�
dens_2/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_5:output:0dens_2_217657dens_2_217659*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_2_layer_call_and_return_conditional_losses_2170722 
dens_2/StatefulPartitionedCall�
dens_1/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_6:output:0dens_1_217662dens_1_217664*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_1_layer_call_and_return_conditional_losses_2170892 
dens_1/StatefulPartitionedCall�
head_7/StatefulPartitionedCallStatefulPartitionedCall'dens_7/StatefulPartitionedCall:output:0head_7_217667head_7_217669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_7_layer_call_and_return_conditional_losses_2171062 
head_7/StatefulPartitionedCall�
head_6/StatefulPartitionedCallStatefulPartitionedCall'dens_6/StatefulPartitionedCall:output:0head_6_217672head_6_217674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_6_layer_call_and_return_conditional_losses_2171232 
head_6/StatefulPartitionedCall�
head_5/StatefulPartitionedCallStatefulPartitionedCall'dens_5/StatefulPartitionedCall:output:0head_5_217677head_5_217679*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_5_layer_call_and_return_conditional_losses_2171402 
head_5/StatefulPartitionedCall�
head_4/StatefulPartitionedCallStatefulPartitionedCall'dens_4/StatefulPartitionedCall:output:0head_4_217682head_4_217684*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_4_layer_call_and_return_conditional_losses_2171572 
head_4/StatefulPartitionedCall�
head_3/StatefulPartitionedCallStatefulPartitionedCall'dens_3/StatefulPartitionedCall:output:0head_3_217687head_3_217689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_3_layer_call_and_return_conditional_losses_2171742 
head_3/StatefulPartitionedCall�
head_2/StatefulPartitionedCallStatefulPartitionedCall'dens_2/StatefulPartitionedCall:output:0head_2_217692head_2_217694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_2_layer_call_and_return_conditional_losses_2171912 
head_2/StatefulPartitionedCall�
head_1/StatefulPartitionedCallStatefulPartitionedCall'dens_1/StatefulPartitionedCall:output:0head_1_217697head_1_217699*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_1_layer_call_and_return_conditional_losses_2172082 
head_1/StatefulPartitionedCall�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__217570*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__217574*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__217578*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentity'head_1/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity'head_2/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity'head_3/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identity'head_4/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identity'head_5/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity'head_6/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identity'head_7/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp2@
dens_1/StatefulPartitionedCalldens_1/StatefulPartitionedCall2@
dens_2/StatefulPartitionedCalldens_2/StatefulPartitionedCall2@
dens_3/StatefulPartitionedCalldens_3/StatefulPartitionedCall2@
dens_4/StatefulPartitionedCalldens_4/StatefulPartitionedCall2@
dens_5/StatefulPartitionedCalldens_5/StatefulPartitionedCall2@
dens_6/StatefulPartitionedCalldens_6/StatefulPartitionedCall2@
dens_7/StatefulPartitionedCalldens_7/StatefulPartitionedCall2@
head_1/StatefulPartitionedCallhead_1/StatefulPartitionedCall2@
head_2/StatefulPartitionedCallhead_2/StatefulPartitionedCall2@
head_3/StatefulPartitionedCallhead_3/StatefulPartitionedCall2@
head_4/StatefulPartitionedCallhead_4/StatefulPartitionedCall2@
head_5/StatefulPartitionedCallhead_5/StatefulPartitionedCall2@
head_6/StatefulPartitionedCallhead_6/StatefulPartitionedCall2@
head_7/StatefulPartitionedCallhead_7/StatefulPartitionedCall2@
trunk_/StatefulPartitionedCalltrunk_/StatefulPartitionedCall2D
 trunk_/StatefulPartitionedCall_1 trunk_/StatefulPartitionedCall_12D
 trunk_/StatefulPartitionedCall_2 trunk_/StatefulPartitionedCall_22D
 trunk_/StatefulPartitionedCall_3 trunk_/StatefulPartitionedCall_32D
 trunk_/StatefulPartitionedCall_4 trunk_/StatefulPartitionedCall_42D
 trunk_/StatefulPartitionedCall_5 trunk_/StatefulPartitionedCall_52D
 trunk_/StatefulPartitionedCall_6 trunk_/StatefulPartitionedCall_6:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dens_3_layer_call_fn_219775

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_3_layer_call_and_return_conditional_losses_2170552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_conv_3_layer_call_and_return_conditional_losses_216504

inputsA
+conv1d_expanddims_1_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������S`2
Relu�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Z@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������Z@
 
_user_specified_nameinputs
�
i
L__inference_global_max_pool__layer_call_and_return_conditional_losses_216889
input_i
identity�
pool_/PartitionedCallPartitionedCallinput_i*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������)`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_pool__layer_call_and_return_conditional_losses_2168302
pool_/PartitionedCall�
flat_/PartitionedCallPartitionedCallpool_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_flat__layer_call_and_return_conditional_losses_2168502
flat_/PartitionedCalls
IdentityIdentityflat_/PartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:T P
+
_output_shapes
:���������S`
!
_user_specified_name	input_I
�
N
1__inference_global_max_pool__layer_call_fn_216883
input_i
identity�
PartitionedCallPartitionedCallinput_i*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:T P
+
_output_shapes
:���������S`
!
_user_specified_name	input_I
�

�
B__inference_dens_1_layer_call_and_return_conditional_losses_217089

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�#
�	
;__inference_multi-task_self-supervised_layer_call_fn_217901
input_1
input_2
input_3
input_4
input_5
input_6
input_7
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@`
	unknown_4:`
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:

unknown_23:	�

unknown_24:

unknown_25:	�

unknown_26:

unknown_27:	�

unknown_28:

unknown_29:	�

unknown_30:

unknown_31:	�

unknown_32:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6input_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*4
Tin-
+2)*
Tout
	2*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������*D
_read_only_resource_inputs&
$"	
 !"#$%&'(*2
config_proto" 

CPU

GPU2*0,1J 8� *_
fZRX
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_2177272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_2:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_3:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_4:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_5:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_6:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_7
�

�
B__inference_head_4_layer_call_and_return_conditional_losses_217157

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�=
!__inference__wrapped_model_216405
input_1
input_2
input_3
input_4
input_5
input_6
input_7j
Tmulti_task_self_supervised_trunk__conv_1_conv1d_expanddims_1_readvariableop_resource: V
Hmulti_task_self_supervised_trunk__conv_1_biasadd_readvariableop_resource: j
Tmulti_task_self_supervised_trunk__conv_2_conv1d_expanddims_1_readvariableop_resource: @V
Hmulti_task_self_supervised_trunk__conv_2_biasadd_readvariableop_resource:@j
Tmulti_task_self_supervised_trunk__conv_3_conv1d_expanddims_1_readvariableop_resource:@`V
Hmulti_task_self_supervised_trunk__conv_3_biasadd_readvariableop_resource:`T
@multi_task_self_supervised_dens_7_matmul_readvariableop_resource:
��P
Amulti_task_self_supervised_dens_7_biasadd_readvariableop_resource:	�T
@multi_task_self_supervised_dens_6_matmul_readvariableop_resource:
��P
Amulti_task_self_supervised_dens_6_biasadd_readvariableop_resource:	�T
@multi_task_self_supervised_dens_5_matmul_readvariableop_resource:
��P
Amulti_task_self_supervised_dens_5_biasadd_readvariableop_resource:	�T
@multi_task_self_supervised_dens_4_matmul_readvariableop_resource:
��P
Amulti_task_self_supervised_dens_4_biasadd_readvariableop_resource:	�T
@multi_task_self_supervised_dens_3_matmul_readvariableop_resource:
��P
Amulti_task_self_supervised_dens_3_biasadd_readvariableop_resource:	�T
@multi_task_self_supervised_dens_2_matmul_readvariableop_resource:
��P
Amulti_task_self_supervised_dens_2_biasadd_readvariableop_resource:	�T
@multi_task_self_supervised_dens_1_matmul_readvariableop_resource:
��P
Amulti_task_self_supervised_dens_1_biasadd_readvariableop_resource:	�S
@multi_task_self_supervised_head_7_matmul_readvariableop_resource:	�O
Amulti_task_self_supervised_head_7_biasadd_readvariableop_resource:S
@multi_task_self_supervised_head_6_matmul_readvariableop_resource:	�O
Amulti_task_self_supervised_head_6_biasadd_readvariableop_resource:S
@multi_task_self_supervised_head_5_matmul_readvariableop_resource:	�O
Amulti_task_self_supervised_head_5_biasadd_readvariableop_resource:S
@multi_task_self_supervised_head_4_matmul_readvariableop_resource:	�O
Amulti_task_self_supervised_head_4_biasadd_readvariableop_resource:S
@multi_task_self_supervised_head_3_matmul_readvariableop_resource:	�O
Amulti_task_self_supervised_head_3_biasadd_readvariableop_resource:S
@multi_task_self_supervised_head_2_matmul_readvariableop_resource:	�O
Amulti_task_self_supervised_head_2_biasadd_readvariableop_resource:S
@multi_task_self_supervised_head_1_matmul_readvariableop_resource:	�O
Amulti_task_self_supervised_head_1_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��8multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp�7multi-task_self-supervised/dens_1/MatMul/ReadVariableOp�8multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp�7multi-task_self-supervised/dens_2/MatMul/ReadVariableOp�8multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp�7multi-task_self-supervised/dens_3/MatMul/ReadVariableOp�8multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp�7multi-task_self-supervised/dens_4/MatMul/ReadVariableOp�8multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp�7multi-task_self-supervised/dens_5/MatMul/ReadVariableOp�8multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp�7multi-task_self-supervised/dens_6/MatMul/ReadVariableOp�8multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp�7multi-task_self-supervised/dens_7/MatMul/ReadVariableOp�8multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp�7multi-task_self-supervised/head_1/MatMul/ReadVariableOp�8multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp�7multi-task_self-supervised/head_2/MatMul/ReadVariableOp�8multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp�7multi-task_self-supervised/head_3/MatMul/ReadVariableOp�8multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp�7multi-task_self-supervised/head_4/MatMul/ReadVariableOp�8multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp�7multi-task_self-supervised/head_5/MatMul/ReadVariableOp�8multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp�7multi-task_self-supervised/head_6/MatMul/ReadVariableOp�8multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp�7multi-task_self-supervised/head_7/MatMul/ReadVariableOp�?multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOp�Kmulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp�?multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOp�Kmulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp�?multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOp�Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOp�Kmulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp�Mmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp�
>multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2@
>multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims/dim�
:multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims
ExpandDimsinput_7Gmulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2<
:multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims�
Kmulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02M
Kmulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp�
@multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/dim�
<multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1
ExpandDimsSmulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Imulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2>
<multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1�
/multi-task_self-supervised/trunk_/conv_1/conv1dConv2DCmulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims:output:0Emulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
21
/multi-task_self-supervised/trunk_/conv_1/conv1d�
7multi-task_self-supervised/trunk_/conv_1/conv1d/SqueezeSqueeze8multi-task_self-supervised/trunk_/conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������29
7multi-task_self-supervised/trunk_/conv_1/conv1d/Squeeze�
?multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02A
?multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOp�
0multi-task_self-supervised/trunk_/conv_1/BiasAddBiasAdd@multi-task_self-supervised/trunk_/conv_1/conv1d/Squeeze:output:0Gmulti-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 22
0multi-task_self-supervised/trunk_/conv_1/BiasAdd�
-multi-task_self-supervised/trunk_/conv_1/ReluRelu9multi-task_self-supervised/trunk_/conv_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������i 2/
-multi-task_self-supervised/trunk_/conv_1/Relu�
1multi-task_self-supervised/trunk_/drop_1/IdentityIdentity;multi-task_self-supervised/trunk_/conv_1/Relu:activations:0*
T0*+
_output_shapes
:���������i 23
1multi-task_self-supervised/trunk_/drop_1/Identity�
>multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2@
>multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims/dim�
:multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims
ExpandDims:multi-task_self-supervised/trunk_/drop_1/Identity:output:0Gmulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2<
:multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims�
Kmulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02M
Kmulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp�
@multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/dim�
<multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1
ExpandDimsSmulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0Imulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2>
<multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1�
/multi-task_self-supervised/trunk_/conv_2/conv1dConv2DCmulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims:output:0Emulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
21
/multi-task_self-supervised/trunk_/conv_2/conv1d�
7multi-task_self-supervised/trunk_/conv_2/conv1d/SqueezeSqueeze8multi-task_self-supervised/trunk_/conv_2/conv1d:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������29
7multi-task_self-supervised/trunk_/conv_2/conv1d/Squeeze�
?multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02A
?multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOp�
0multi-task_self-supervised/trunk_/conv_2/BiasAddBiasAdd@multi-task_self-supervised/trunk_/conv_2/conv1d/Squeeze:output:0Gmulti-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@22
0multi-task_self-supervised/trunk_/conv_2/BiasAdd�
-multi-task_self-supervised/trunk_/conv_2/ReluRelu9multi-task_self-supervised/trunk_/conv_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������Z@2/
-multi-task_self-supervised/trunk_/conv_2/Relu�
1multi-task_self-supervised/trunk_/drop_2/IdentityIdentity;multi-task_self-supervised/trunk_/conv_2/Relu:activations:0*
T0*+
_output_shapes
:���������Z@23
1multi-task_self-supervised/trunk_/drop_2/Identity�
>multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2@
>multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims/dim�
:multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims
ExpandDims:multi-task_self-supervised/trunk_/drop_2/Identity:output:0Gmulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2<
:multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims�
Kmulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02M
Kmulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp�
@multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/dim�
<multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1
ExpandDimsSmulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0Imulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2>
<multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1�
/multi-task_self-supervised/trunk_/conv_3/conv1dConv2DCmulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims:output:0Emulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
21
/multi-task_self-supervised/trunk_/conv_3/conv1d�
7multi-task_self-supervised/trunk_/conv_3/conv1d/SqueezeSqueeze8multi-task_self-supervised/trunk_/conv_3/conv1d:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������29
7multi-task_self-supervised/trunk_/conv_3/conv1d/Squeeze�
?multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02A
?multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOp�
0multi-task_self-supervised/trunk_/conv_3/BiasAddBiasAdd@multi-task_self-supervised/trunk_/conv_3/conv1d/Squeeze:output:0Gmulti-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`22
0multi-task_self-supervised/trunk_/conv_3/BiasAdd�
-multi-task_self-supervised/trunk_/conv_3/ReluRelu9multi-task_self-supervised/trunk_/conv_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������S`2/
-multi-task_self-supervised/trunk_/conv_3/Relu�
1multi-task_self-supervised/trunk_/drop_3/IdentityIdentity;multi-task_self-supervised/trunk_/conv_3/Relu:activations:0*
T0*+
_output_shapes
:���������S`23
1multi-task_self-supervised/trunk_/drop_3/Identity�
@multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims
ExpandDimsinput_6Imulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2>
<multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02O
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2@
>multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_1/conv1d_1Conv2DEmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_1/conv1d_1�
9multi-task_self-supervised/trunk_/conv_1/conv1d_1/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_1/conv1d_1/Squeeze�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_1BiasAddBmulti-task_self-supervised/trunk_/conv_1/conv1d_1/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 24
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_1�
/multi-task_self-supervised/trunk_/conv_1/Relu_1Relu;multi-task_self-supervised/trunk_/conv_1/BiasAdd_1:output:0*
T0*+
_output_shapes
:���������i 21
/multi-task_self-supervised/trunk_/conv_1/Relu_1�
3multi-task_self-supervised/trunk_/drop_1/Identity_1Identity=multi-task_self-supervised/trunk_/conv_1/Relu_1:activations:0*
T0*+
_output_shapes
:���������i 25
3multi-task_self-supervised/trunk_/drop_1/Identity_1�
@multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_1/Identity_1:output:0Imulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2>
<multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2@
>multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_2/conv1d_1Conv2DEmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_2/conv1d_1�
9multi-task_self-supervised/trunk_/conv_2/conv1d_1/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_2/conv1d_1:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_2/conv1d_1/Squeeze�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_1BiasAddBmulti-task_self-supervised/trunk_/conv_2/conv1d_1/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@24
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_1�
/multi-task_self-supervised/trunk_/conv_2/Relu_1Relu;multi-task_self-supervised/trunk_/conv_2/BiasAdd_1:output:0*
T0*+
_output_shapes
:���������Z@21
/multi-task_self-supervised/trunk_/conv_2/Relu_1�
3multi-task_self-supervised/trunk_/drop_2/Identity_1Identity=multi-task_self-supervised/trunk_/conv_2/Relu_1:activations:0*
T0*+
_output_shapes
:���������Z@25
3multi-task_self-supervised/trunk_/drop_2/Identity_1�
@multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_2/Identity_1:output:0Imulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2>
<multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2@
>multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_3/conv1d_1Conv2DEmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_3/conv1d_1�
9multi-task_self-supervised/trunk_/conv_3/conv1d_1/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_3/conv1d_1:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_3/conv1d_1/Squeeze�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02C
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_1BiasAddBmulti-task_self-supervised/trunk_/conv_3/conv1d_1/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`24
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_1�
/multi-task_self-supervised/trunk_/conv_3/Relu_1Relu;multi-task_self-supervised/trunk_/conv_3/BiasAdd_1:output:0*
T0*+
_output_shapes
:���������S`21
/multi-task_self-supervised/trunk_/conv_3/Relu_1�
3multi-task_self-supervised/trunk_/drop_3/Identity_1Identity=multi-task_self-supervised/trunk_/conv_3/Relu_1:activations:0*
T0*+
_output_shapes
:���������S`25
3multi-task_self-supervised/trunk_/drop_3/Identity_1�
@multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims
ExpandDimsinput_5Imulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2>
<multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02O
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2@
>multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_1/conv1d_2Conv2DEmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_1/conv1d_2�
9multi-task_self-supervised/trunk_/conv_1/conv1d_2/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_1/conv1d_2:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_1/conv1d_2/Squeeze�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_2BiasAddBmulti-task_self-supervised/trunk_/conv_1/conv1d_2/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 24
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_2�
/multi-task_self-supervised/trunk_/conv_1/Relu_2Relu;multi-task_self-supervised/trunk_/conv_1/BiasAdd_2:output:0*
T0*+
_output_shapes
:���������i 21
/multi-task_self-supervised/trunk_/conv_1/Relu_2�
3multi-task_self-supervised/trunk_/drop_1/Identity_2Identity=multi-task_self-supervised/trunk_/conv_1/Relu_2:activations:0*
T0*+
_output_shapes
:���������i 25
3multi-task_self-supervised/trunk_/drop_1/Identity_2�
@multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_1/Identity_2:output:0Imulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2>
<multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2@
>multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_2/conv1d_2Conv2DEmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_2/conv1d_2�
9multi-task_self-supervised/trunk_/conv_2/conv1d_2/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_2/conv1d_2:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_2/conv1d_2/Squeeze�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_2BiasAddBmulti-task_self-supervised/trunk_/conv_2/conv1d_2/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@24
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_2�
/multi-task_self-supervised/trunk_/conv_2/Relu_2Relu;multi-task_self-supervised/trunk_/conv_2/BiasAdd_2:output:0*
T0*+
_output_shapes
:���������Z@21
/multi-task_self-supervised/trunk_/conv_2/Relu_2�
3multi-task_self-supervised/trunk_/drop_2/Identity_2Identity=multi-task_self-supervised/trunk_/conv_2/Relu_2:activations:0*
T0*+
_output_shapes
:���������Z@25
3multi-task_self-supervised/trunk_/drop_2/Identity_2�
@multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_2/Identity_2:output:0Imulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2>
<multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2@
>multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_3/conv1d_2Conv2DEmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_3/conv1d_2�
9multi-task_self-supervised/trunk_/conv_3/conv1d_2/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_3/conv1d_2:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_3/conv1d_2/Squeeze�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02C
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_2BiasAddBmulti-task_self-supervised/trunk_/conv_3/conv1d_2/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`24
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_2�
/multi-task_self-supervised/trunk_/conv_3/Relu_2Relu;multi-task_self-supervised/trunk_/conv_3/BiasAdd_2:output:0*
T0*+
_output_shapes
:���������S`21
/multi-task_self-supervised/trunk_/conv_3/Relu_2�
3multi-task_self-supervised/trunk_/drop_3/Identity_2Identity=multi-task_self-supervised/trunk_/conv_3/Relu_2:activations:0*
T0*+
_output_shapes
:���������S`25
3multi-task_self-supervised/trunk_/drop_3/Identity_2�
@multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims
ExpandDimsinput_4Imulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2>
<multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02O
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2@
>multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_1/conv1d_3Conv2DEmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_1/conv1d_3�
9multi-task_self-supervised/trunk_/conv_1/conv1d_3/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_1/conv1d_3:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_1/conv1d_3/Squeeze�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_3BiasAddBmulti-task_self-supervised/trunk_/conv_1/conv1d_3/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 24
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_3�
/multi-task_self-supervised/trunk_/conv_1/Relu_3Relu;multi-task_self-supervised/trunk_/conv_1/BiasAdd_3:output:0*
T0*+
_output_shapes
:���������i 21
/multi-task_self-supervised/trunk_/conv_1/Relu_3�
3multi-task_self-supervised/trunk_/drop_1/Identity_3Identity=multi-task_self-supervised/trunk_/conv_1/Relu_3:activations:0*
T0*+
_output_shapes
:���������i 25
3multi-task_self-supervised/trunk_/drop_1/Identity_3�
@multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_1/Identity_3:output:0Imulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2>
<multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2@
>multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_2/conv1d_3Conv2DEmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_2/conv1d_3�
9multi-task_self-supervised/trunk_/conv_2/conv1d_3/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_2/conv1d_3:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_2/conv1d_3/Squeeze�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_3BiasAddBmulti-task_self-supervised/trunk_/conv_2/conv1d_3/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@24
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_3�
/multi-task_self-supervised/trunk_/conv_2/Relu_3Relu;multi-task_self-supervised/trunk_/conv_2/BiasAdd_3:output:0*
T0*+
_output_shapes
:���������Z@21
/multi-task_self-supervised/trunk_/conv_2/Relu_3�
3multi-task_self-supervised/trunk_/drop_2/Identity_3Identity=multi-task_self-supervised/trunk_/conv_2/Relu_3:activations:0*
T0*+
_output_shapes
:���������Z@25
3multi-task_self-supervised/trunk_/drop_2/Identity_3�
@multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_2/Identity_3:output:0Imulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2>
<multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2@
>multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_3/conv1d_3Conv2DEmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_3/conv1d_3�
9multi-task_self-supervised/trunk_/conv_3/conv1d_3/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_3/conv1d_3:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_3/conv1d_3/Squeeze�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02C
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_3BiasAddBmulti-task_self-supervised/trunk_/conv_3/conv1d_3/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`24
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_3�
/multi-task_self-supervised/trunk_/conv_3/Relu_3Relu;multi-task_self-supervised/trunk_/conv_3/BiasAdd_3:output:0*
T0*+
_output_shapes
:���������S`21
/multi-task_self-supervised/trunk_/conv_3/Relu_3�
3multi-task_self-supervised/trunk_/drop_3/Identity_3Identity=multi-task_self-supervised/trunk_/conv_3/Relu_3:activations:0*
T0*+
_output_shapes
:���������S`25
3multi-task_self-supervised/trunk_/drop_3/Identity_3�
@multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims
ExpandDimsinput_3Imulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2>
<multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02O
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2@
>multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_1/conv1d_4Conv2DEmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_1/conv1d_4�
9multi-task_self-supervised/trunk_/conv_1/conv1d_4/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_1/conv1d_4:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_1/conv1d_4/Squeeze�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_4BiasAddBmulti-task_self-supervised/trunk_/conv_1/conv1d_4/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 24
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_4�
/multi-task_self-supervised/trunk_/conv_1/Relu_4Relu;multi-task_self-supervised/trunk_/conv_1/BiasAdd_4:output:0*
T0*+
_output_shapes
:���������i 21
/multi-task_self-supervised/trunk_/conv_1/Relu_4�
3multi-task_self-supervised/trunk_/drop_1/Identity_4Identity=multi-task_self-supervised/trunk_/conv_1/Relu_4:activations:0*
T0*+
_output_shapes
:���������i 25
3multi-task_self-supervised/trunk_/drop_1/Identity_4�
@multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_1/Identity_4:output:0Imulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2>
<multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2@
>multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_2/conv1d_4Conv2DEmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_2/conv1d_4�
9multi-task_self-supervised/trunk_/conv_2/conv1d_4/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_2/conv1d_4:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_2/conv1d_4/Squeeze�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_4BiasAddBmulti-task_self-supervised/trunk_/conv_2/conv1d_4/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@24
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_4�
/multi-task_self-supervised/trunk_/conv_2/Relu_4Relu;multi-task_self-supervised/trunk_/conv_2/BiasAdd_4:output:0*
T0*+
_output_shapes
:���������Z@21
/multi-task_self-supervised/trunk_/conv_2/Relu_4�
3multi-task_self-supervised/trunk_/drop_2/Identity_4Identity=multi-task_self-supervised/trunk_/conv_2/Relu_4:activations:0*
T0*+
_output_shapes
:���������Z@25
3multi-task_self-supervised/trunk_/drop_2/Identity_4�
@multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_2/Identity_4:output:0Imulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2>
<multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2@
>multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_3/conv1d_4Conv2DEmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_3/conv1d_4�
9multi-task_self-supervised/trunk_/conv_3/conv1d_4/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_3/conv1d_4:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_3/conv1d_4/Squeeze�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02C
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_4BiasAddBmulti-task_self-supervised/trunk_/conv_3/conv1d_4/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`24
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_4�
/multi-task_self-supervised/trunk_/conv_3/Relu_4Relu;multi-task_self-supervised/trunk_/conv_3/BiasAdd_4:output:0*
T0*+
_output_shapes
:���������S`21
/multi-task_self-supervised/trunk_/conv_3/Relu_4�
3multi-task_self-supervised/trunk_/drop_3/Identity_4Identity=multi-task_self-supervised/trunk_/conv_3/Relu_4:activations:0*
T0*+
_output_shapes
:���������S`25
3multi-task_self-supervised/trunk_/drop_3/Identity_4�
@multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims
ExpandDimsinput_2Imulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2>
<multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02O
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2@
>multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_1/conv1d_5Conv2DEmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_1/conv1d_5�
9multi-task_self-supervised/trunk_/conv_1/conv1d_5/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_1/conv1d_5:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_1/conv1d_5/Squeeze�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_5BiasAddBmulti-task_self-supervised/trunk_/conv_1/conv1d_5/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 24
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_5�
/multi-task_self-supervised/trunk_/conv_1/Relu_5Relu;multi-task_self-supervised/trunk_/conv_1/BiasAdd_5:output:0*
T0*+
_output_shapes
:���������i 21
/multi-task_self-supervised/trunk_/conv_1/Relu_5�
3multi-task_self-supervised/trunk_/drop_1/Identity_5Identity=multi-task_self-supervised/trunk_/conv_1/Relu_5:activations:0*
T0*+
_output_shapes
:���������i 25
3multi-task_self-supervised/trunk_/drop_1/Identity_5�
@multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_1/Identity_5:output:0Imulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2>
<multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2@
>multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_2/conv1d_5Conv2DEmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_2/conv1d_5�
9multi-task_self-supervised/trunk_/conv_2/conv1d_5/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_2/conv1d_5:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_2/conv1d_5/Squeeze�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_5BiasAddBmulti-task_self-supervised/trunk_/conv_2/conv1d_5/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@24
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_5�
/multi-task_self-supervised/trunk_/conv_2/Relu_5Relu;multi-task_self-supervised/trunk_/conv_2/BiasAdd_5:output:0*
T0*+
_output_shapes
:���������Z@21
/multi-task_self-supervised/trunk_/conv_2/Relu_5�
3multi-task_self-supervised/trunk_/drop_2/Identity_5Identity=multi-task_self-supervised/trunk_/conv_2/Relu_5:activations:0*
T0*+
_output_shapes
:���������Z@25
3multi-task_self-supervised/trunk_/drop_2/Identity_5�
@multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_2/Identity_5:output:0Imulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2>
<multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2@
>multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_3/conv1d_5Conv2DEmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_3/conv1d_5�
9multi-task_self-supervised/trunk_/conv_3/conv1d_5/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_3/conv1d_5:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_3/conv1d_5/Squeeze�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02C
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_5BiasAddBmulti-task_self-supervised/trunk_/conv_3/conv1d_5/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`24
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_5�
/multi-task_self-supervised/trunk_/conv_3/Relu_5Relu;multi-task_self-supervised/trunk_/conv_3/BiasAdd_5:output:0*
T0*+
_output_shapes
:���������S`21
/multi-task_self-supervised/trunk_/conv_3/Relu_5�
3multi-task_self-supervised/trunk_/drop_3/Identity_5Identity=multi-task_self-supervised/trunk_/conv_3/Relu_5:activations:0*
T0*+
_output_shapes
:���������S`25
3multi-task_self-supervised/trunk_/drop_3/Identity_5�
@multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims
ExpandDimsinput_1Imulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2>
<multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02O
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2@
>multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_1/conv1d_6Conv2DEmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_1/conv1d_6�
9multi-task_self-supervised/trunk_/conv_1/conv1d_6/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_1/conv1d_6:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_1/conv1d_6/Squeeze�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_6BiasAddBmulti-task_self-supervised/trunk_/conv_1/conv1d_6/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 24
2multi-task_self-supervised/trunk_/conv_1/BiasAdd_6�
/multi-task_self-supervised/trunk_/conv_1/Relu_6Relu;multi-task_self-supervised/trunk_/conv_1/BiasAdd_6:output:0*
T0*+
_output_shapes
:���������i 21
/multi-task_self-supervised/trunk_/conv_1/Relu_6�
3multi-task_self-supervised/trunk_/drop_1/Identity_6Identity=multi-task_self-supervised/trunk_/conv_1/Relu_6:activations:0*
T0*+
_output_shapes
:���������i 25
3multi-task_self-supervised/trunk_/drop_1/Identity_6�
@multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_1/Identity_6:output:0Imulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2>
<multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2@
>multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_2/conv1d_6Conv2DEmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_2/conv1d_6�
9multi-task_self-supervised/trunk_/conv_2/conv1d_6/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_2/conv1d_6:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_2/conv1d_6/Squeeze�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_6BiasAddBmulti-task_self-supervised/trunk_/conv_2/conv1d_6/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@24
2multi-task_self-supervised/trunk_/conv_2/BiasAdd_6�
/multi-task_self-supervised/trunk_/conv_2/Relu_6Relu;multi-task_self-supervised/trunk_/conv_2/BiasAdd_6:output:0*
T0*+
_output_shapes
:���������Z@21
/multi-task_self-supervised/trunk_/conv_2/Relu_6�
3multi-task_self-supervised/trunk_/drop_2/Identity_6Identity=multi-task_self-supervised/trunk_/conv_2/Relu_6:activations:0*
T0*+
_output_shapes
:���������Z@25
3multi-task_self-supervised/trunk_/drop_2/Identity_6�
@multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2B
@multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims/dim�
<multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims
ExpandDims<multi-task_self-supervised/trunk_/drop_2/Identity_6:output:0Imulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2>
<multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOpReadVariableOpTmulti_task_self_supervised_trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02O
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp�
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/dim�
>multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1
ExpandDimsUmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp:value:0Kmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2@
>multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1�
1multi-task_self-supervised/trunk_/conv_3/conv1d_6Conv2DEmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims:output:0Gmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
23
1multi-task_self-supervised/trunk_/conv_3/conv1d_6�
9multi-task_self-supervised/trunk_/conv_3/conv1d_6/SqueezeSqueeze:multi-task_self-supervised/trunk_/conv_3/conv1d_6:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2;
9multi-task_self-supervised/trunk_/conv_3/conv1d_6/Squeeze�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOpReadVariableOpHmulti_task_self_supervised_trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02C
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOp�
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_6BiasAddBmulti-task_self-supervised/trunk_/conv_3/conv1d_6/Squeeze:output:0Imulti-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`24
2multi-task_self-supervised/trunk_/conv_3/BiasAdd_6�
/multi-task_self-supervised/trunk_/conv_3/Relu_6Relu;multi-task_self-supervised/trunk_/conv_3/BiasAdd_6:output:0*
T0*+
_output_shapes
:���������S`21
/multi-task_self-supervised/trunk_/conv_3/Relu_6�
3multi-task_self-supervised/trunk_/drop_3/Identity_6Identity=multi-task_self-supervised/trunk_/conv_3/Relu_6:activations:0*
T0*+
_output_shapes
:���������S`25
3multi-task_self-supervised/trunk_/drop_3/Identity_6�
@multi-task_self-supervised/global_max_pool_/pool_/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2B
@multi-task_self-supervised/global_max_pool_/pool_/ExpandDims/dim�
<multi-task_self-supervised/global_max_pool_/pool_/ExpandDims
ExpandDims:multi-task_self-supervised/trunk_/drop_3/Identity:output:0Imulti-task_self-supervised/global_max_pool_/pool_/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������S`2>
<multi-task_self-supervised/global_max_pool_/pool_/ExpandDims�
9multi-task_self-supervised/global_max_pool_/pool_/MaxPoolMaxPoolEmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2;
9multi-task_self-supervised/global_max_pool_/pool_/MaxPool�
9multi-task_self-supervised/global_max_pool_/pool_/SqueezeSqueezeBmulti-task_self-supervised/global_max_pool_/pool_/MaxPool:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2;
9multi-task_self-supervised/global_max_pool_/pool_/Squeeze�
7multi-task_self-supervised/global_max_pool_/flat_/ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  29
7multi-task_self-supervised/global_max_pool_/flat_/Const�
9multi-task_self-supervised/global_max_pool_/flat_/ReshapeReshapeBmulti-task_self-supervised/global_max_pool_/pool_/Squeeze:output:0@multi-task_self-supervised/global_max_pool_/flat_/Const:output:0*
T0*(
_output_shapes
:����������2;
9multi-task_self-supervised/global_max_pool_/flat_/Reshape�
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2D
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_1/dim�
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_1
ExpandDims<multi-task_self-supervised/trunk_/drop_3/Identity_1:output:0Kmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:���������S`2@
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_1�
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_1MaxPoolGmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_1:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2=
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_1�
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_1SqueezeDmulti-task_self-supervised/global_max_pool_/pool_/MaxPool_1:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2=
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_1�
9multi-task_self-supervised/global_max_pool_/flat_/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����`  2;
9multi-task_self-supervised/global_max_pool_/flat_/Const_1�
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_1ReshapeDmulti-task_self-supervised/global_max_pool_/pool_/Squeeze_1:output:0Bmulti-task_self-supervised/global_max_pool_/flat_/Const_1:output:0*
T0*(
_output_shapes
:����������2=
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_1�
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2D
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_2/dim�
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_2
ExpandDims<multi-task_self-supervised/trunk_/drop_3/Identity_2:output:0Kmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_2/dim:output:0*
T0*/
_output_shapes
:���������S`2@
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_2�
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_2MaxPoolGmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_2:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2=
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_2�
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_2SqueezeDmulti-task_self-supervised/global_max_pool_/pool_/MaxPool_2:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2=
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_2�
9multi-task_self-supervised/global_max_pool_/flat_/Const_2Const*
_output_shapes
:*
dtype0*
valueB"����`  2;
9multi-task_self-supervised/global_max_pool_/flat_/Const_2�
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_2ReshapeDmulti-task_self-supervised/global_max_pool_/pool_/Squeeze_2:output:0Bmulti-task_self-supervised/global_max_pool_/flat_/Const_2:output:0*
T0*(
_output_shapes
:����������2=
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_2�
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :2D
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_3/dim�
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_3
ExpandDims<multi-task_self-supervised/trunk_/drop_3/Identity_3:output:0Kmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:���������S`2@
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_3�
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_3MaxPoolGmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_3:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2=
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_3�
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_3SqueezeDmulti-task_self-supervised/global_max_pool_/pool_/MaxPool_3:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2=
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_3�
9multi-task_self-supervised/global_max_pool_/flat_/Const_3Const*
_output_shapes
:*
dtype0*
valueB"����`  2;
9multi-task_self-supervised/global_max_pool_/flat_/Const_3�
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_3ReshapeDmulti-task_self-supervised/global_max_pool_/pool_/Squeeze_3:output:0Bmulti-task_self-supervised/global_max_pool_/flat_/Const_3:output:0*
T0*(
_output_shapes
:����������2=
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_3�
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :2D
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_4/dim�
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_4
ExpandDims<multi-task_self-supervised/trunk_/drop_3/Identity_4:output:0Kmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_4/dim:output:0*
T0*/
_output_shapes
:���������S`2@
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_4�
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_4MaxPoolGmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_4:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2=
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_4�
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_4SqueezeDmulti-task_self-supervised/global_max_pool_/pool_/MaxPool_4:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2=
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_4�
9multi-task_self-supervised/global_max_pool_/flat_/Const_4Const*
_output_shapes
:*
dtype0*
valueB"����`  2;
9multi-task_self-supervised/global_max_pool_/flat_/Const_4�
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_4ReshapeDmulti-task_self-supervised/global_max_pool_/pool_/Squeeze_4:output:0Bmulti-task_self-supervised/global_max_pool_/flat_/Const_4:output:0*
T0*(
_output_shapes
:����������2=
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_4�
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :2D
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_5/dim�
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_5
ExpandDims<multi-task_self-supervised/trunk_/drop_3/Identity_5:output:0Kmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:���������S`2@
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_5�
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_5MaxPoolGmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_5:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2=
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_5�
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_5SqueezeDmulti-task_self-supervised/global_max_pool_/pool_/MaxPool_5:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2=
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_5�
9multi-task_self-supervised/global_max_pool_/flat_/Const_5Const*
_output_shapes
:*
dtype0*
valueB"����`  2;
9multi-task_self-supervised/global_max_pool_/flat_/Const_5�
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_5ReshapeDmulti-task_self-supervised/global_max_pool_/pool_/Squeeze_5:output:0Bmulti-task_self-supervised/global_max_pool_/flat_/Const_5:output:0*
T0*(
_output_shapes
:����������2=
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_5�
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :2D
Bmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_6/dim�
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_6
ExpandDims<multi-task_self-supervised/trunk_/drop_3/Identity_6:output:0Kmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:���������S`2@
>multi-task_self-supervised/global_max_pool_/pool_/ExpandDims_6�
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_6MaxPoolGmulti-task_self-supervised/global_max_pool_/pool_/ExpandDims_6:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2=
;multi-task_self-supervised/global_max_pool_/pool_/MaxPool_6�
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_6SqueezeDmulti-task_self-supervised/global_max_pool_/pool_/MaxPool_6:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2=
;multi-task_self-supervised/global_max_pool_/pool_/Squeeze_6�
9multi-task_self-supervised/global_max_pool_/flat_/Const_6Const*
_output_shapes
:*
dtype0*
valueB"����`  2;
9multi-task_self-supervised/global_max_pool_/flat_/Const_6�
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_6ReshapeDmulti-task_self-supervised/global_max_pool_/pool_/Squeeze_6:output:0Bmulti-task_self-supervised/global_max_pool_/flat_/Const_6:output:0*
T0*(
_output_shapes
:����������2=
;multi-task_self-supervised/global_max_pool_/flat_/Reshape_6�
7multi-task_self-supervised/dens_7/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_dens_7_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7multi-task_self-supervised/dens_7/MatMul/ReadVariableOp�
(multi-task_self-supervised/dens_7/MatMulMatMulBmulti-task_self-supervised/global_max_pool_/flat_/Reshape:output:0?multi-task_self-supervised/dens_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(multi-task_self-supervised/dens_7/MatMul�
8multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_dens_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/dens_7/BiasAddBiasAdd2multi-task_self-supervised/dens_7/MatMul:product:0@multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)multi-task_self-supervised/dens_7/BiasAdd�
&multi-task_self-supervised/dens_7/ReluRelu2multi-task_self-supervised/dens_7/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&multi-task_self-supervised/dens_7/Relu�
7multi-task_self-supervised/dens_6/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_dens_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7multi-task_self-supervised/dens_6/MatMul/ReadVariableOp�
(multi-task_self-supervised/dens_6/MatMulMatMulDmulti-task_self-supervised/global_max_pool_/flat_/Reshape_1:output:0?multi-task_self-supervised/dens_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(multi-task_self-supervised/dens_6/MatMul�
8multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_dens_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/dens_6/BiasAddBiasAdd2multi-task_self-supervised/dens_6/MatMul:product:0@multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)multi-task_self-supervised/dens_6/BiasAdd�
&multi-task_self-supervised/dens_6/ReluRelu2multi-task_self-supervised/dens_6/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&multi-task_self-supervised/dens_6/Relu�
7multi-task_self-supervised/dens_5/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_dens_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7multi-task_self-supervised/dens_5/MatMul/ReadVariableOp�
(multi-task_self-supervised/dens_5/MatMulMatMulDmulti-task_self-supervised/global_max_pool_/flat_/Reshape_2:output:0?multi-task_self-supervised/dens_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(multi-task_self-supervised/dens_5/MatMul�
8multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_dens_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/dens_5/BiasAddBiasAdd2multi-task_self-supervised/dens_5/MatMul:product:0@multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)multi-task_self-supervised/dens_5/BiasAdd�
&multi-task_self-supervised/dens_5/ReluRelu2multi-task_self-supervised/dens_5/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&multi-task_self-supervised/dens_5/Relu�
7multi-task_self-supervised/dens_4/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_dens_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7multi-task_self-supervised/dens_4/MatMul/ReadVariableOp�
(multi-task_self-supervised/dens_4/MatMulMatMulDmulti-task_self-supervised/global_max_pool_/flat_/Reshape_3:output:0?multi-task_self-supervised/dens_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(multi-task_self-supervised/dens_4/MatMul�
8multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_dens_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/dens_4/BiasAddBiasAdd2multi-task_self-supervised/dens_4/MatMul:product:0@multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)multi-task_self-supervised/dens_4/BiasAdd�
&multi-task_self-supervised/dens_4/ReluRelu2multi-task_self-supervised/dens_4/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&multi-task_self-supervised/dens_4/Relu�
7multi-task_self-supervised/dens_3/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_dens_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7multi-task_self-supervised/dens_3/MatMul/ReadVariableOp�
(multi-task_self-supervised/dens_3/MatMulMatMulDmulti-task_self-supervised/global_max_pool_/flat_/Reshape_4:output:0?multi-task_self-supervised/dens_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(multi-task_self-supervised/dens_3/MatMul�
8multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_dens_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/dens_3/BiasAddBiasAdd2multi-task_self-supervised/dens_3/MatMul:product:0@multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)multi-task_self-supervised/dens_3/BiasAdd�
&multi-task_self-supervised/dens_3/ReluRelu2multi-task_self-supervised/dens_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&multi-task_self-supervised/dens_3/Relu�
7multi-task_self-supervised/dens_2/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_dens_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7multi-task_self-supervised/dens_2/MatMul/ReadVariableOp�
(multi-task_self-supervised/dens_2/MatMulMatMulDmulti-task_self-supervised/global_max_pool_/flat_/Reshape_5:output:0?multi-task_self-supervised/dens_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(multi-task_self-supervised/dens_2/MatMul�
8multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_dens_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/dens_2/BiasAddBiasAdd2multi-task_self-supervised/dens_2/MatMul:product:0@multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)multi-task_self-supervised/dens_2/BiasAdd�
&multi-task_self-supervised/dens_2/ReluRelu2multi-task_self-supervised/dens_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&multi-task_self-supervised/dens_2/Relu�
7multi-task_self-supervised/dens_1/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_dens_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7multi-task_self-supervised/dens_1/MatMul/ReadVariableOp�
(multi-task_self-supervised/dens_1/MatMulMatMulDmulti-task_self-supervised/global_max_pool_/flat_/Reshape_6:output:0?multi-task_self-supervised/dens_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(multi-task_self-supervised/dens_1/MatMul�
8multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_dens_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/dens_1/BiasAddBiasAdd2multi-task_self-supervised/dens_1/MatMul:product:0@multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)multi-task_self-supervised/dens_1/BiasAdd�
&multi-task_self-supervised/dens_1/ReluRelu2multi-task_self-supervised/dens_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&multi-task_self-supervised/dens_1/Relu�
7multi-task_self-supervised/head_7/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_head_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7multi-task_self-supervised/head_7/MatMul/ReadVariableOp�
(multi-task_self-supervised/head_7/MatMulMatMul4multi-task_self-supervised/dens_7/Relu:activations:0?multi-task_self-supervised/head_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2*
(multi-task_self-supervised/head_7/MatMul�
8multi-task_self-supervised/head_7/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_head_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/head_7/BiasAddBiasAdd2multi-task_self-supervised/head_7/MatMul:product:0@multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_7/BiasAdd�
)multi-task_self-supervised/head_7/SigmoidSigmoid2multi-task_self-supervised/head_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_7/Sigmoid�
7multi-task_self-supervised/head_6/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_head_6_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7multi-task_self-supervised/head_6/MatMul/ReadVariableOp�
(multi-task_self-supervised/head_6/MatMulMatMul4multi-task_self-supervised/dens_6/Relu:activations:0?multi-task_self-supervised/head_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2*
(multi-task_self-supervised/head_6/MatMul�
8multi-task_self-supervised/head_6/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_head_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/head_6/BiasAddBiasAdd2multi-task_self-supervised/head_6/MatMul:product:0@multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_6/BiasAdd�
)multi-task_self-supervised/head_6/SigmoidSigmoid2multi-task_self-supervised/head_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_6/Sigmoid�
7multi-task_self-supervised/head_5/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_head_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7multi-task_self-supervised/head_5/MatMul/ReadVariableOp�
(multi-task_self-supervised/head_5/MatMulMatMul4multi-task_self-supervised/dens_5/Relu:activations:0?multi-task_self-supervised/head_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2*
(multi-task_self-supervised/head_5/MatMul�
8multi-task_self-supervised/head_5/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_head_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/head_5/BiasAddBiasAdd2multi-task_self-supervised/head_5/MatMul:product:0@multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_5/BiasAdd�
)multi-task_self-supervised/head_5/SigmoidSigmoid2multi-task_self-supervised/head_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_5/Sigmoid�
7multi-task_self-supervised/head_4/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_head_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7multi-task_self-supervised/head_4/MatMul/ReadVariableOp�
(multi-task_self-supervised/head_4/MatMulMatMul4multi-task_self-supervised/dens_4/Relu:activations:0?multi-task_self-supervised/head_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2*
(multi-task_self-supervised/head_4/MatMul�
8multi-task_self-supervised/head_4/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_head_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/head_4/BiasAddBiasAdd2multi-task_self-supervised/head_4/MatMul:product:0@multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_4/BiasAdd�
)multi-task_self-supervised/head_4/SigmoidSigmoid2multi-task_self-supervised/head_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_4/Sigmoid�
7multi-task_self-supervised/head_3/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_head_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7multi-task_self-supervised/head_3/MatMul/ReadVariableOp�
(multi-task_self-supervised/head_3/MatMulMatMul4multi-task_self-supervised/dens_3/Relu:activations:0?multi-task_self-supervised/head_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2*
(multi-task_self-supervised/head_3/MatMul�
8multi-task_self-supervised/head_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_head_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/head_3/BiasAddBiasAdd2multi-task_self-supervised/head_3/MatMul:product:0@multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_3/BiasAdd�
)multi-task_self-supervised/head_3/SigmoidSigmoid2multi-task_self-supervised/head_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_3/Sigmoid�
7multi-task_self-supervised/head_2/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_head_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7multi-task_self-supervised/head_2/MatMul/ReadVariableOp�
(multi-task_self-supervised/head_2/MatMulMatMul4multi-task_self-supervised/dens_2/Relu:activations:0?multi-task_self-supervised/head_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2*
(multi-task_self-supervised/head_2/MatMul�
8multi-task_self-supervised/head_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_head_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/head_2/BiasAddBiasAdd2multi-task_self-supervised/head_2/MatMul:product:0@multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_2/BiasAdd�
)multi-task_self-supervised/head_2/SigmoidSigmoid2multi-task_self-supervised/head_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_2/Sigmoid�
7multi-task_self-supervised/head_1/MatMul/ReadVariableOpReadVariableOp@multi_task_self_supervised_head_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7multi-task_self-supervised/head_1/MatMul/ReadVariableOp�
(multi-task_self-supervised/head_1/MatMulMatMul4multi-task_self-supervised/dens_1/Relu:activations:0?multi-task_self-supervised/head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2*
(multi-task_self-supervised/head_1/MatMul�
8multi-task_self-supervised/head_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_task_self_supervised_head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp�
)multi-task_self-supervised/head_1/BiasAddBiasAdd2multi-task_self-supervised/head_1/MatMul:product:0@multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_1/BiasAdd�
)multi-task_self-supervised/head_1/SigmoidSigmoid2multi-task_self-supervised/head_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2+
)multi-task_self-supervised/head_1/Sigmoid�%
IdentityIdentity-multi-task_self-supervised/head_1/Sigmoid:y:09^multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_1/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_2/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_3/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_4/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_5/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_6/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_7/MatMul/ReadVariableOp9^multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_1/MatMul/ReadVariableOp9^multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_2/MatMul/ReadVariableOp9^multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_3/MatMul/ReadVariableOp9^multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_4/MatMul/ReadVariableOp9^multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_5/MatMul/ReadVariableOp9^multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_6/MatMul/ReadVariableOp9^multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_7/MatMul/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�&

Identity_1Identity-multi-task_self-supervised/head_2/Sigmoid:y:09^multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_1/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_2/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_3/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_4/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_5/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_6/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_7/MatMul/ReadVariableOp9^multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_1/MatMul/ReadVariableOp9^multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_2/MatMul/ReadVariableOp9^multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_3/MatMul/ReadVariableOp9^multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_4/MatMul/ReadVariableOp9^multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_5/MatMul/ReadVariableOp9^multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_6/MatMul/ReadVariableOp9^multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_7/MatMul/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�&

Identity_2Identity-multi-task_self-supervised/head_3/Sigmoid:y:09^multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_1/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_2/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_3/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_4/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_5/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_6/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_7/MatMul/ReadVariableOp9^multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_1/MatMul/ReadVariableOp9^multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_2/MatMul/ReadVariableOp9^multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_3/MatMul/ReadVariableOp9^multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_4/MatMul/ReadVariableOp9^multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_5/MatMul/ReadVariableOp9^multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_6/MatMul/ReadVariableOp9^multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_7/MatMul/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2�&

Identity_3Identity-multi-task_self-supervised/head_4/Sigmoid:y:09^multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_1/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_2/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_3/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_4/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_5/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_6/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_7/MatMul/ReadVariableOp9^multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_1/MatMul/ReadVariableOp9^multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_2/MatMul/ReadVariableOp9^multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_3/MatMul/ReadVariableOp9^multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_4/MatMul/ReadVariableOp9^multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_5/MatMul/ReadVariableOp9^multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_6/MatMul/ReadVariableOp9^multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_7/MatMul/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_3�&

Identity_4Identity-multi-task_self-supervised/head_5/Sigmoid:y:09^multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_1/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_2/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_3/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_4/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_5/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_6/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_7/MatMul/ReadVariableOp9^multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_1/MatMul/ReadVariableOp9^multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_2/MatMul/ReadVariableOp9^multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_3/MatMul/ReadVariableOp9^multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_4/MatMul/ReadVariableOp9^multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_5/MatMul/ReadVariableOp9^multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_6/MatMul/ReadVariableOp9^multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_7/MatMul/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�&

Identity_5Identity-multi-task_self-supervised/head_6/Sigmoid:y:09^multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_1/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_2/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_3/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_4/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_5/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_6/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_7/MatMul/ReadVariableOp9^multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_1/MatMul/ReadVariableOp9^multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_2/MatMul/ReadVariableOp9^multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_3/MatMul/ReadVariableOp9^multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_4/MatMul/ReadVariableOp9^multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_5/MatMul/ReadVariableOp9^multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_6/MatMul/ReadVariableOp9^multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_7/MatMul/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5�&

Identity_6Identity-multi-task_self-supervised/head_7/Sigmoid:y:09^multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_1/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_2/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_3/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_4/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_5/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_6/MatMul/ReadVariableOp9^multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/dens_7/MatMul/ReadVariableOp9^multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_1/MatMul/ReadVariableOp9^multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_2/MatMul/ReadVariableOp9^multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_3/MatMul/ReadVariableOp9^multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_4/MatMul/ReadVariableOp9^multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_5/MatMul/ReadVariableOp9^multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_6/MatMul/ReadVariableOp9^multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp8^multi-task_self-supervised/head_7/MatMul/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp@^multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOpB^multi-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOpL^multi-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpN^multi-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp8multi-task_self-supervised/dens_1/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/dens_1/MatMul/ReadVariableOp7multi-task_self-supervised/dens_1/MatMul/ReadVariableOp2t
8multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp8multi-task_self-supervised/dens_2/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/dens_2/MatMul/ReadVariableOp7multi-task_self-supervised/dens_2/MatMul/ReadVariableOp2t
8multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp8multi-task_self-supervised/dens_3/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/dens_3/MatMul/ReadVariableOp7multi-task_self-supervised/dens_3/MatMul/ReadVariableOp2t
8multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp8multi-task_self-supervised/dens_4/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/dens_4/MatMul/ReadVariableOp7multi-task_self-supervised/dens_4/MatMul/ReadVariableOp2t
8multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp8multi-task_self-supervised/dens_5/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/dens_5/MatMul/ReadVariableOp7multi-task_self-supervised/dens_5/MatMul/ReadVariableOp2t
8multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp8multi-task_self-supervised/dens_6/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/dens_6/MatMul/ReadVariableOp7multi-task_self-supervised/dens_6/MatMul/ReadVariableOp2t
8multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp8multi-task_self-supervised/dens_7/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/dens_7/MatMul/ReadVariableOp7multi-task_self-supervised/dens_7/MatMul/ReadVariableOp2t
8multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp8multi-task_self-supervised/head_1/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/head_1/MatMul/ReadVariableOp7multi-task_self-supervised/head_1/MatMul/ReadVariableOp2t
8multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp8multi-task_self-supervised/head_2/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/head_2/MatMul/ReadVariableOp7multi-task_self-supervised/head_2/MatMul/ReadVariableOp2t
8multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp8multi-task_self-supervised/head_3/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/head_3/MatMul/ReadVariableOp7multi-task_self-supervised/head_3/MatMul/ReadVariableOp2t
8multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp8multi-task_self-supervised/head_4/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/head_4/MatMul/ReadVariableOp7multi-task_self-supervised/head_4/MatMul/ReadVariableOp2t
8multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp8multi-task_self-supervised/head_5/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/head_5/MatMul/ReadVariableOp7multi-task_self-supervised/head_5/MatMul/ReadVariableOp2t
8multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp8multi-task_self-supervised/head_6/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/head_6/MatMul/ReadVariableOp7multi-task_self-supervised/head_6/MatMul/ReadVariableOp2t
8multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp8multi-task_self-supervised/head_7/BiasAdd/ReadVariableOp2r
7multi-task_self-supervised/head_7/MatMul/ReadVariableOp7multi-task_self-supervised/head_7/MatMul/ReadVariableOp2�
?multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOp?multi-task_self-supervised/trunk_/conv_1/BiasAdd/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_1/BiasAdd_1/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_1/BiasAdd_2/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_1/BiasAdd_3/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_1/BiasAdd_4/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_1/BiasAdd_5/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_1/BiasAdd_6/ReadVariableOp2�
Kmulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpKmulti-task_self-supervised/trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp2�
?multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOp?multi-task_self-supervised/trunk_/conv_2/BiasAdd/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_2/BiasAdd_1/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_2/BiasAdd_2/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_2/BiasAdd_3/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_2/BiasAdd_4/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_2/BiasAdd_5/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_2/BiasAdd_6/ReadVariableOp2�
Kmulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpKmulti-task_self-supervised/trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp2�
?multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOp?multi-task_self-supervised/trunk_/conv_3/BiasAdd/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_3/BiasAdd_1/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_3/BiasAdd_2/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_3/BiasAdd_3/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_3/BiasAdd_4/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_3/BiasAdd_5/ReadVariableOp2�
Amulti-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOpAmulti-task_self-supervised/trunk_/conv_3/BiasAdd_6/ReadVariableOp2�
Kmulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpKmulti-task_self-supervised/trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp2�
Mmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOpMmulti-task_self-supervised/trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_2:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_3:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_4:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_5:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_6:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_7
�

�
B__inference_dens_6_layer_call_and_return_conditional_losses_219846

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dens_4_layer_call_and_return_conditional_losses_217038

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dens_5_layer_call_fn_219815

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_5_layer_call_and_return_conditional_losses_2170212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_head_1_layer_call_and_return_conditional_losses_219886

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
B__inference_drop_3_layer_call_and_return_conditional_losses_220198

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������S`2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������S`2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�

�
B__inference_head_4_layer_call_and_return_conditional_losses_219946

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
B__inference_drop_3_layer_call_and_return_conditional_losses_216515

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������S`2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������S`2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�
a
B__inference_drop_2_layer_call_and_return_conditional_losses_216604

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Z@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z@:S O
+
_output_shapes
:���������Z@
 
_user_specified_nameinputs
�
�
'__inference_head_2_layer_call_fn_219895

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_2_layer_call_and_return_conditional_losses_2171912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
i
L__inference_global_max_pool__layer_call_and_return_conditional_losses_216895
input_i
identity�
pool_/PartitionedCallPartitionedCallinput_i*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������)`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_pool__layer_call_and_return_conditional_losses_2168302
pool_/PartitionedCall�
flat_/PartitionedCallPartitionedCallpool_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_flat__layer_call_and_return_conditional_losses_2168502
flat_/PartitionedCalls
IdentityIdentityflat_/PartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:T P
+
_output_shapes
:���������S`
!
_user_specified_name	input_I
�

�
B__inference_dens_1_layer_call_and_return_conditional_losses_219746

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dens_7_layer_call_and_return_conditional_losses_216987

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
B__inference_drop_3_layer_call_and_return_conditional_losses_220186

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������S`2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������S`2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
��
�
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_217239

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6#
trunk__216914: 
trunk__216916: #
trunk__216918: @
trunk__216920:@#
trunk__216922:@`
trunk__216924:`!
dens_7_216988:
��
dens_7_216990:	�!
dens_6_217005:
��
dens_6_217007:	�!
dens_5_217022:
��
dens_5_217024:	�!
dens_4_217039:
��
dens_4_217041:	�!
dens_3_217056:
��
dens_3_217058:	�!
dens_2_217073:
��
dens_2_217075:	�!
dens_1_217090:
��
dens_1_217092:	� 
head_7_217107:	�
head_7_217109: 
head_6_217124:	�
head_6_217126: 
head_5_217141:	�
head_5_217143: 
head_4_217158:	�
head_4_217160: 
head_3_217175:	�
head_3_217177: 
head_2_217192:	�
head_2_217194: 
head_1_217209:	�
head_1_217211:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��/conv_1/kernel/Regularizer/Square/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOp�dens_1/StatefulPartitionedCall�dens_2/StatefulPartitionedCall�dens_3/StatefulPartitionedCall�dens_4/StatefulPartitionedCall�dens_5/StatefulPartitionedCall�dens_6/StatefulPartitionedCall�dens_7/StatefulPartitionedCall�head_1/StatefulPartitionedCall�head_2/StatefulPartitionedCall�head_3/StatefulPartitionedCall�head_4/StatefulPartitionedCall�head_5/StatefulPartitionedCall�head_6/StatefulPartitionedCall�head_7/StatefulPartitionedCall�trunk_/StatefulPartitionedCall� trunk_/StatefulPartitionedCall_1� trunk_/StatefulPartitionedCall_2� trunk_/StatefulPartitionedCall_3� trunk_/StatefulPartitionedCall_4� trunk_/StatefulPartitionedCall_5� trunk_/StatefulPartitionedCall_6�
trunk_/StatefulPartitionedCallStatefulPartitionedCallinputs_6trunk__216914trunk__216916trunk__216918trunk__216920trunk__216922trunk__216924*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362 
trunk_/StatefulPartitionedCall�
 trunk_/StatefulPartitionedCall_1StatefulPartitionedCallinputs_5trunk__216914trunk__216916trunk__216918trunk__216920trunk__216922trunk__216924*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_1�
 trunk_/StatefulPartitionedCall_2StatefulPartitionedCallinputs_4trunk__216914trunk__216916trunk__216918trunk__216920trunk__216922trunk__216924*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_2�
 trunk_/StatefulPartitionedCall_3StatefulPartitionedCallinputs_3trunk__216914trunk__216916trunk__216918trunk__216920trunk__216922trunk__216924*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_3�
 trunk_/StatefulPartitionedCall_4StatefulPartitionedCallinputs_2trunk__216914trunk__216916trunk__216918trunk__216920trunk__216922trunk__216924*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_4�
 trunk_/StatefulPartitionedCall_5StatefulPartitionedCallinputs_1trunk__216914trunk__216916trunk__216918trunk__216920trunk__216922trunk__216924*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_5�
 trunk_/StatefulPartitionedCall_6StatefulPartitionedCallinputstrunk__216914trunk__216916trunk__216918trunk__216920trunk__216922trunk__216924*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362"
 trunk_/StatefulPartitionedCall_6�
 global_max_pool_/PartitionedCallPartitionedCall'trunk_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532"
 global_max_pool_/PartitionedCall�
"global_max_pool_/PartitionedCall_1PartitionedCall)trunk_/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_1�
"global_max_pool_/PartitionedCall_2PartitionedCall)trunk_/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_2�
"global_max_pool_/PartitionedCall_3PartitionedCall)trunk_/StatefulPartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_3�
"global_max_pool_/PartitionedCall_4PartitionedCall)trunk_/StatefulPartitionedCall_4:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_4�
"global_max_pool_/PartitionedCall_5PartitionedCall)trunk_/StatefulPartitionedCall_5:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_5�
"global_max_pool_/PartitionedCall_6PartitionedCall)trunk_/StatefulPartitionedCall_6:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532$
"global_max_pool_/PartitionedCall_6�
dens_7/StatefulPartitionedCallStatefulPartitionedCall)global_max_pool_/PartitionedCall:output:0dens_7_216988dens_7_216990*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_7_layer_call_and_return_conditional_losses_2169872 
dens_7/StatefulPartitionedCall�
dens_6/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_1:output:0dens_6_217005dens_6_217007*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_6_layer_call_and_return_conditional_losses_2170042 
dens_6/StatefulPartitionedCall�
dens_5/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_2:output:0dens_5_217022dens_5_217024*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_5_layer_call_and_return_conditional_losses_2170212 
dens_5/StatefulPartitionedCall�
dens_4/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_3:output:0dens_4_217039dens_4_217041*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_4_layer_call_and_return_conditional_losses_2170382 
dens_4/StatefulPartitionedCall�
dens_3/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_4:output:0dens_3_217056dens_3_217058*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_3_layer_call_and_return_conditional_losses_2170552 
dens_3/StatefulPartitionedCall�
dens_2/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_5:output:0dens_2_217073dens_2_217075*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_2_layer_call_and_return_conditional_losses_2170722 
dens_2/StatefulPartitionedCall�
dens_1/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_6:output:0dens_1_217090dens_1_217092*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_1_layer_call_and_return_conditional_losses_2170892 
dens_1/StatefulPartitionedCall�
head_7/StatefulPartitionedCallStatefulPartitionedCall'dens_7/StatefulPartitionedCall:output:0head_7_217107head_7_217109*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_7_layer_call_and_return_conditional_losses_2171062 
head_7/StatefulPartitionedCall�
head_6/StatefulPartitionedCallStatefulPartitionedCall'dens_6/StatefulPartitionedCall:output:0head_6_217124head_6_217126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_6_layer_call_and_return_conditional_losses_2171232 
head_6/StatefulPartitionedCall�
head_5/StatefulPartitionedCallStatefulPartitionedCall'dens_5/StatefulPartitionedCall:output:0head_5_217141head_5_217143*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_5_layer_call_and_return_conditional_losses_2171402 
head_5/StatefulPartitionedCall�
head_4/StatefulPartitionedCallStatefulPartitionedCall'dens_4/StatefulPartitionedCall:output:0head_4_217158head_4_217160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_4_layer_call_and_return_conditional_losses_2171572 
head_4/StatefulPartitionedCall�
head_3/StatefulPartitionedCallStatefulPartitionedCall'dens_3/StatefulPartitionedCall:output:0head_3_217175head_3_217177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_3_layer_call_and_return_conditional_losses_2171742 
head_3/StatefulPartitionedCall�
head_2/StatefulPartitionedCallStatefulPartitionedCall'dens_2/StatefulPartitionedCall:output:0head_2_217192head_2_217194*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_2_layer_call_and_return_conditional_losses_2171912 
head_2/StatefulPartitionedCall�
head_1/StatefulPartitionedCallStatefulPartitionedCall'dens_1/StatefulPartitionedCall:output:0head_1_217209head_1_217211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_1_layer_call_and_return_conditional_losses_2172082 
head_1/StatefulPartitionedCall�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__216914*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__216918*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__216922*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentity'head_1/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity'head_2/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity'head_3/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identity'head_4/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identity'head_5/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity'head_6/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identity'head_7/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp2@
dens_1/StatefulPartitionedCalldens_1/StatefulPartitionedCall2@
dens_2/StatefulPartitionedCalldens_2/StatefulPartitionedCall2@
dens_3/StatefulPartitionedCalldens_3/StatefulPartitionedCall2@
dens_4/StatefulPartitionedCalldens_4/StatefulPartitionedCall2@
dens_5/StatefulPartitionedCalldens_5/StatefulPartitionedCall2@
dens_6/StatefulPartitionedCalldens_6/StatefulPartitionedCall2@
dens_7/StatefulPartitionedCalldens_7/StatefulPartitionedCall2@
head_1/StatefulPartitionedCallhead_1/StatefulPartitionedCall2@
head_2/StatefulPartitionedCallhead_2/StatefulPartitionedCall2@
head_3/StatefulPartitionedCallhead_3/StatefulPartitionedCall2@
head_4/StatefulPartitionedCallhead_4/StatefulPartitionedCall2@
head_5/StatefulPartitionedCallhead_5/StatefulPartitionedCall2@
head_6/StatefulPartitionedCallhead_6/StatefulPartitionedCall2@
head_7/StatefulPartitionedCallhead_7/StatefulPartitionedCall2@
trunk_/StatefulPartitionedCalltrunk_/StatefulPartitionedCall2D
 trunk_/StatefulPartitionedCall_1 trunk_/StatefulPartitionedCall_12D
 trunk_/StatefulPartitionedCall_2 trunk_/StatefulPartitionedCall_22D
 trunk_/StatefulPartitionedCall_3 trunk_/StatefulPartitionedCall_32D
 trunk_/StatefulPartitionedCall_4 trunk_/StatefulPartitionedCall_42D
 trunk_/StatefulPartitionedCall_5 trunk_/StatefulPartitionedCall_52D
 trunk_/StatefulPartitionedCall_6 trunk_/StatefulPartitionedCall_6:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������
 
_user_specified_nameinputs
�#
�	
;__inference_multi-task_self-supervised_layer_call_fn_218441
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@`
	unknown_4:`
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:

unknown_23:	�

unknown_24:

unknown_25:	�

unknown_26:

unknown_27:	�

unknown_28:

unknown_29:	�

unknown_30:

unknown_31:	�

unknown_32:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*4
Tin-
+2)*
Tout
	2*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������*D
_read_only_resource_inputs&
$"	
 !"#$%&'(*2
config_proto" 

CPU

GPU2*0,1J 8� *_
fZRX
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_2172392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/2:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/3:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/4:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/5:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/6
�
�
'__inference_trunk__layer_call_fn_219553

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@`
	unknown_4:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�<
�
B__inference_trunk__layer_call_and_return_conditional_losses_216821

input_#
conv_1_216784: 
conv_1_216786: #
conv_2_216790: @
conv_2_216792:@#
conv_3_216796:@`
conv_3_216798:`
identity��conv_1/StatefulPartitionedCall�/conv_1/kernel/Regularizer/Square/ReadVariableOp�conv_2/StatefulPartitionedCall�/conv_2/kernel/Regularizer/Square/ReadVariableOp�conv_3/StatefulPartitionedCall�/conv_3/kernel/Regularizer/Square/ReadVariableOp�drop_1/StatefulPartitionedCall�drop_2/StatefulPartitionedCall�drop_3/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_conv_1_216784conv_1_216786*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_2164342 
conv_1/StatefulPartitionedCall�
drop_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_1_layer_call_and_return_conditional_losses_2166372 
drop_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'drop_1/StatefulPartitionedCall:output:0conv_2_216790conv_2_216792*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_2164692 
conv_2/StatefulPartitionedCall�
drop_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0^drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_2_layer_call_and_return_conditional_losses_2166042 
drop_2/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall'drop_2/StatefulPartitionedCall:output:0conv_3_216796conv_3_216798*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_2165042 
conv_3/StatefulPartitionedCall�
drop_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0^drop_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_2165712 
drop_3/StatefulPartitionedCall�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_1_216784*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_2_216790*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_3_216796*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentity'drop_3/StatefulPartitionedCall:output:0^conv_1/StatefulPartitionedCall0^conv_1/kernel/Regularizer/Square/ReadVariableOp^conv_2/StatefulPartitionedCall0^conv_2/kernel/Regularizer/Square/ReadVariableOp^conv_3/StatefulPartitionedCall0^conv_3/kernel/Regularizer/Square/ReadVariableOp^drop_1/StatefulPartitionedCall^drop_2/StatefulPartitionedCall^drop_3/StatefulPartitionedCall*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp2@
drop_1/StatefulPartitionedCalldrop_1/StatefulPartitionedCall2@
drop_2/StatefulPartitionedCalldrop_2/StatefulPartitionedCall2@
drop_3/StatefulPartitionedCalldrop_3/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinput_
�
C
'__inference_drop_3_layer_call_fn_220176

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_2165152
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�
�
'__inference_head_1_layer_call_fn_219875

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_1_layer_call_and_return_conditional_losses_2172082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_head_7_layer_call_fn_219995

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_7_layer_call_and_return_conditional_losses_2171062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_head_2_layer_call_and_return_conditional_losses_219906

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
'__inference_drop_1_layer_call_fn_220053

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_1_layer_call_and_return_conditional_losses_2166372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������i 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������i 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������i 
 
_user_specified_nameinputs
�#
�	
;__inference_multi-task_self-supervised_layer_call_fn_217322
input_1
input_2
input_3
input_4
input_5
input_6
input_7
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@`
	unknown_4:`
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:

unknown_23:	�

unknown_24:

unknown_25:	�

unknown_26:

unknown_27:	�

unknown_28:

unknown_29:	�

unknown_30:

unknown_31:	�

unknown_32:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6input_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*4
Tin-
+2)*
Tout
	2*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������*D
_read_only_resource_inputs&
$"	
 !"#$%&'(*2
config_proto" 

CPU

GPU2*0,1J 8� *_
fZRX
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_2172392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_2:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_3:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_4:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_5:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_6:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_7
�
�
__inference_loss_fn_1_220220N
8conv_2_kernel_regularizer_square_readvariableop_resource: @
identity��/conv_2/kernel/Regularizer/Square/ReadVariableOp�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv_2_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
IdentityIdentity!conv_2/kernel/Regularizer/mul:z:00^conv_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp
�
�
'__inference_head_3_layer_call_fn_219915

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_3_layer_call_and_return_conditional_losses_2171742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
]
A__inference_flat__layer_call_and_return_conditional_losses_216850

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������)`:S O
+
_output_shapes
:���������)`
 
_user_specified_nameinputs
�
C
'__inference_drop_1_layer_call_fn_220048

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_1_layer_call_and_return_conditional_losses_2164452
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������i 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������i :S O
+
_output_shapes
:���������i 
 
_user_specified_nameinputs
�

�
B__inference_head_2_layer_call_and_return_conditional_losses_217191

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_head_6_layer_call_fn_219975

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_6_layer_call_and_return_conditional_losses_2171232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dens_5_layer_call_and_return_conditional_losses_217021

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dens_5_layer_call_and_return_conditional_losses_219826

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
]
A__inference_pool__layer_call_and_return_conditional_losses_216830

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
C
'__inference_drop_2_layer_call_fn_220112

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_2_layer_call_and_return_conditional_losses_2164802
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z@:S O
+
_output_shapes
:���������Z@
 
_user_specified_nameinputs
�
`
'__inference_drop_2_layer_call_fn_220117

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_2_layer_call_and_return_conditional_losses_2166042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Z@
 
_user_specified_nameinputs
�
`
B__inference_drop_2_layer_call_and_return_conditional_losses_220122

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������Z@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������Z@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z@:S O
+
_output_shapes
:���������Z@
 
_user_specified_nameinputs
�

�
B__inference_dens_3_layer_call_and_return_conditional_losses_217055

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_head_5_layer_call_and_return_conditional_losses_217140

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
B
&__inference_pool__layer_call_fn_216836

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_pool__layer_call_and_return_conditional_losses_2168302
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
N
1__inference_global_max_pool__layer_call_fn_216856
input_i
identity�
PartitionedCallPartitionedCallinput_i*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:T P
+
_output_shapes
:���������S`
!
_user_specified_name	input_I
�
�
B__inference_conv_2_layer_call_and_return_conditional_losses_216469

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������Z@2
Relu�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������i : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������i 
 
_user_specified_nameinputs
�
a
B__inference_drop_1_layer_call_and_return_conditional_losses_216637

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������i 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������i 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������i 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������i :S O
+
_output_shapes
:���������i 
 
_user_specified_nameinputs
�
`
'__inference_drop_3_layer_call_fn_220181

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_2165712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
��
�
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_218233
input_1
input_2
input_3
input_4
input_5
input_6
input_7#
trunk__218076: 
trunk__218078: #
trunk__218080: @
trunk__218082:@#
trunk__218084:@`
trunk__218086:`!
dens_7_218138:
��
dens_7_218140:	�!
dens_6_218143:
��
dens_6_218145:	�!
dens_5_218148:
��
dens_5_218150:	�!
dens_4_218153:
��
dens_4_218155:	�!
dens_3_218158:
��
dens_3_218160:	�!
dens_2_218163:
��
dens_2_218165:	�!
dens_1_218168:
��
dens_1_218170:	� 
head_7_218173:	�
head_7_218175: 
head_6_218178:	�
head_6_218180: 
head_5_218183:	�
head_5_218185: 
head_4_218188:	�
head_4_218190: 
head_3_218193:	�
head_3_218195: 
head_2_218198:	�
head_2_218200: 
head_1_218203:	�
head_1_218205:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��/conv_1/kernel/Regularizer/Square/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOp�dens_1/StatefulPartitionedCall�dens_2/StatefulPartitionedCall�dens_3/StatefulPartitionedCall�dens_4/StatefulPartitionedCall�dens_5/StatefulPartitionedCall�dens_6/StatefulPartitionedCall�dens_7/StatefulPartitionedCall�head_1/StatefulPartitionedCall�head_2/StatefulPartitionedCall�head_3/StatefulPartitionedCall�head_4/StatefulPartitionedCall�head_5/StatefulPartitionedCall�head_6/StatefulPartitionedCall�head_7/StatefulPartitionedCall�trunk_/StatefulPartitionedCall� trunk_/StatefulPartitionedCall_1� trunk_/StatefulPartitionedCall_2� trunk_/StatefulPartitionedCall_3� trunk_/StatefulPartitionedCall_4� trunk_/StatefulPartitionedCall_5� trunk_/StatefulPartitionedCall_6�
trunk_/StatefulPartitionedCallStatefulPartitionedCallinput_7trunk__218076trunk__218078trunk__218080trunk__218082trunk__218084trunk__218086*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092 
trunk_/StatefulPartitionedCall�
 trunk_/StatefulPartitionedCall_1StatefulPartitionedCallinput_6trunk__218076trunk__218078trunk__218080trunk__218082trunk__218084trunk__218086*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_1�
 trunk_/StatefulPartitionedCall_2StatefulPartitionedCallinput_5trunk__218076trunk__218078trunk__218080trunk__218082trunk__218084trunk__218086*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_2�
 trunk_/StatefulPartitionedCall_3StatefulPartitionedCallinput_4trunk__218076trunk__218078trunk__218080trunk__218082trunk__218084trunk__218086*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_3�
 trunk_/StatefulPartitionedCall_4StatefulPartitionedCallinput_3trunk__218076trunk__218078trunk__218080trunk__218082trunk__218084trunk__218086*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_4�
 trunk_/StatefulPartitionedCall_5StatefulPartitionedCallinput_2trunk__218076trunk__218078trunk__218080trunk__218082trunk__218084trunk__218086*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_5�
 trunk_/StatefulPartitionedCall_6StatefulPartitionedCallinput_1trunk__218076trunk__218078trunk__218080trunk__218082trunk__218084trunk__218086*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092"
 trunk_/StatefulPartitionedCall_6�
 global_max_pool_/PartitionedCallPartitionedCall'trunk_/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752"
 global_max_pool_/PartitionedCall�
"global_max_pool_/PartitionedCall_1PartitionedCall)trunk_/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_1�
"global_max_pool_/PartitionedCall_2PartitionedCall)trunk_/StatefulPartitionedCall_2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_2�
"global_max_pool_/PartitionedCall_3PartitionedCall)trunk_/StatefulPartitionedCall_3:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_3�
"global_max_pool_/PartitionedCall_4PartitionedCall)trunk_/StatefulPartitionedCall_4:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_4�
"global_max_pool_/PartitionedCall_5PartitionedCall)trunk_/StatefulPartitionedCall_5:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_5�
"global_max_pool_/PartitionedCall_6PartitionedCall)trunk_/StatefulPartitionedCall_6:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752$
"global_max_pool_/PartitionedCall_6�
dens_7/StatefulPartitionedCallStatefulPartitionedCall)global_max_pool_/PartitionedCall:output:0dens_7_218138dens_7_218140*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_7_layer_call_and_return_conditional_losses_2169872 
dens_7/StatefulPartitionedCall�
dens_6/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_1:output:0dens_6_218143dens_6_218145*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_6_layer_call_and_return_conditional_losses_2170042 
dens_6/StatefulPartitionedCall�
dens_5/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_2:output:0dens_5_218148dens_5_218150*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_5_layer_call_and_return_conditional_losses_2170212 
dens_5/StatefulPartitionedCall�
dens_4/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_3:output:0dens_4_218153dens_4_218155*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_4_layer_call_and_return_conditional_losses_2170382 
dens_4/StatefulPartitionedCall�
dens_3/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_4:output:0dens_3_218158dens_3_218160*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_3_layer_call_and_return_conditional_losses_2170552 
dens_3/StatefulPartitionedCall�
dens_2/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_5:output:0dens_2_218163dens_2_218165*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_2_layer_call_and_return_conditional_losses_2170722 
dens_2/StatefulPartitionedCall�
dens_1/StatefulPartitionedCallStatefulPartitionedCall+global_max_pool_/PartitionedCall_6:output:0dens_1_218168dens_1_218170*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_1_layer_call_and_return_conditional_losses_2170892 
dens_1/StatefulPartitionedCall�
head_7/StatefulPartitionedCallStatefulPartitionedCall'dens_7/StatefulPartitionedCall:output:0head_7_218173head_7_218175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_7_layer_call_and_return_conditional_losses_2171062 
head_7/StatefulPartitionedCall�
head_6/StatefulPartitionedCallStatefulPartitionedCall'dens_6/StatefulPartitionedCall:output:0head_6_218178head_6_218180*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_6_layer_call_and_return_conditional_losses_2171232 
head_6/StatefulPartitionedCall�
head_5/StatefulPartitionedCallStatefulPartitionedCall'dens_5/StatefulPartitionedCall:output:0head_5_218183head_5_218185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_5_layer_call_and_return_conditional_losses_2171402 
head_5/StatefulPartitionedCall�
head_4/StatefulPartitionedCallStatefulPartitionedCall'dens_4/StatefulPartitionedCall:output:0head_4_218188head_4_218190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_4_layer_call_and_return_conditional_losses_2171572 
head_4/StatefulPartitionedCall�
head_3/StatefulPartitionedCallStatefulPartitionedCall'dens_3/StatefulPartitionedCall:output:0head_3_218193head_3_218195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_3_layer_call_and_return_conditional_losses_2171742 
head_3/StatefulPartitionedCall�
head_2/StatefulPartitionedCallStatefulPartitionedCall'dens_2/StatefulPartitionedCall:output:0head_2_218198head_2_218200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_2_layer_call_and_return_conditional_losses_2171912 
head_2/StatefulPartitionedCall�
head_1/StatefulPartitionedCallStatefulPartitionedCall'dens_1/StatefulPartitionedCall:output:0head_1_218203head_1_218205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_1_layer_call_and_return_conditional_losses_2172082 
head_1/StatefulPartitionedCall�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__218076*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__218080*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOptrunk__218084*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentity'head_1/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity'head_2/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity'head_3/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identity'head_4/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identity'head_5/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity'head_6/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identity'head_7/StatefulPartitionedCall:output:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/StatefulPartitionedCall^dens_2/StatefulPartitionedCall^dens_3/StatefulPartitionedCall^dens_4/StatefulPartitionedCall^dens_5/StatefulPartitionedCall^dens_6/StatefulPartitionedCall^dens_7/StatefulPartitionedCall^head_1/StatefulPartitionedCall^head_2/StatefulPartitionedCall^head_3/StatefulPartitionedCall^head_4/StatefulPartitionedCall^head_5/StatefulPartitionedCall^head_6/StatefulPartitionedCall^head_7/StatefulPartitionedCall^trunk_/StatefulPartitionedCall!^trunk_/StatefulPartitionedCall_1!^trunk_/StatefulPartitionedCall_2!^trunk_/StatefulPartitionedCall_3!^trunk_/StatefulPartitionedCall_4!^trunk_/StatefulPartitionedCall_5!^trunk_/StatefulPartitionedCall_6*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp2@
dens_1/StatefulPartitionedCalldens_1/StatefulPartitionedCall2@
dens_2/StatefulPartitionedCalldens_2/StatefulPartitionedCall2@
dens_3/StatefulPartitionedCalldens_3/StatefulPartitionedCall2@
dens_4/StatefulPartitionedCalldens_4/StatefulPartitionedCall2@
dens_5/StatefulPartitionedCalldens_5/StatefulPartitionedCall2@
dens_6/StatefulPartitionedCalldens_6/StatefulPartitionedCall2@
dens_7/StatefulPartitionedCalldens_7/StatefulPartitionedCall2@
head_1/StatefulPartitionedCallhead_1/StatefulPartitionedCall2@
head_2/StatefulPartitionedCallhead_2/StatefulPartitionedCall2@
head_3/StatefulPartitionedCallhead_3/StatefulPartitionedCall2@
head_4/StatefulPartitionedCallhead_4/StatefulPartitionedCall2@
head_5/StatefulPartitionedCallhead_5/StatefulPartitionedCall2@
head_6/StatefulPartitionedCallhead_6/StatefulPartitionedCall2@
head_7/StatefulPartitionedCallhead_7/StatefulPartitionedCall2@
trunk_/StatefulPartitionedCalltrunk_/StatefulPartitionedCall2D
 trunk_/StatefulPartitionedCall_1 trunk_/StatefulPartitionedCall_12D
 trunk_/StatefulPartitionedCall_2 trunk_/StatefulPartitionedCall_22D
 trunk_/StatefulPartitionedCall_3 trunk_/StatefulPartitionedCall_32D
 trunk_/StatefulPartitionedCall_4 trunk_/StatefulPartitionedCall_42D
 trunk_/StatefulPartitionedCall_5 trunk_/StatefulPartitionedCall_52D
 trunk_/StatefulPartitionedCall_6 trunk_/StatefulPartitionedCall_6:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_2:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_3:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_4:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_5:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_6:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_7
�

�
B__inference_head_7_layer_call_and_return_conditional_losses_217106

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
B__inference_drop_1_layer_call_and_return_conditional_losses_220070

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������i 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������i 2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������i 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������i :S O
+
_output_shapes
:���������i 
 
_user_specified_nameinputs
�

�
B__inference_head_6_layer_call_and_return_conditional_losses_217123

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_head_7_layer_call_and_return_conditional_losses_220006

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_head_4_layer_call_fn_219935

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_4_layer_call_and_return_conditional_losses_2171572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�3
__inference__traced_save_220688
file_prefix,
(savev2_dens_1_kernel_read_readvariableop*
&savev2_dens_1_bias_read_readvariableop,
(savev2_dens_2_kernel_read_readvariableop*
&savev2_dens_2_bias_read_readvariableop,
(savev2_dens_3_kernel_read_readvariableop*
&savev2_dens_3_bias_read_readvariableop,
(savev2_dens_4_kernel_read_readvariableop*
&savev2_dens_4_bias_read_readvariableop,
(savev2_dens_5_kernel_read_readvariableop*
&savev2_dens_5_bias_read_readvariableop,
(savev2_dens_6_kernel_read_readvariableop*
&savev2_dens_6_bias_read_readvariableop,
(savev2_dens_7_kernel_read_readvariableop*
&savev2_dens_7_bias_read_readvariableop,
(savev2_head_1_kernel_read_readvariableop*
&savev2_head_1_bias_read_readvariableop,
(savev2_head_2_kernel_read_readvariableop*
&savev2_head_2_bias_read_readvariableop,
(savev2_head_3_kernel_read_readvariableop*
&savev2_head_3_bias_read_readvariableop,
(savev2_head_4_kernel_read_readvariableop*
&savev2_head_4_bias_read_readvariableop,
(savev2_head_5_kernel_read_readvariableop*
&savev2_head_5_bias_read_readvariableop,
(savev2_head_6_kernel_read_readvariableop*
&savev2_head_6_bias_read_readvariableop,
(savev2_head_7_kernel_read_readvariableop*
&savev2_head_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableop&
"savev2_total_7_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_8_read_readvariableop&
"savev2_count_8_read_readvariableop&
"savev2_total_9_read_readvariableop&
"savev2_count_9_read_readvariableop'
#savev2_total_10_read_readvariableop'
#savev2_count_10_read_readvariableop'
#savev2_total_11_read_readvariableop'
#savev2_count_11_read_readvariableop'
#savev2_total_12_read_readvariableop'
#savev2_count_12_read_readvariableop'
#savev2_total_13_read_readvariableop'
#savev2_count_13_read_readvariableop'
#savev2_total_14_read_readvariableop'
#savev2_count_14_read_readvariableop3
/savev2_adam_dens_1_kernel_m_read_readvariableop1
-savev2_adam_dens_1_bias_m_read_readvariableop3
/savev2_adam_dens_2_kernel_m_read_readvariableop1
-savev2_adam_dens_2_bias_m_read_readvariableop3
/savev2_adam_dens_3_kernel_m_read_readvariableop1
-savev2_adam_dens_3_bias_m_read_readvariableop3
/savev2_adam_dens_4_kernel_m_read_readvariableop1
-savev2_adam_dens_4_bias_m_read_readvariableop3
/savev2_adam_dens_5_kernel_m_read_readvariableop1
-savev2_adam_dens_5_bias_m_read_readvariableop3
/savev2_adam_dens_6_kernel_m_read_readvariableop1
-savev2_adam_dens_6_bias_m_read_readvariableop3
/savev2_adam_dens_7_kernel_m_read_readvariableop1
-savev2_adam_dens_7_bias_m_read_readvariableop3
/savev2_adam_head_1_kernel_m_read_readvariableop1
-savev2_adam_head_1_bias_m_read_readvariableop3
/savev2_adam_head_2_kernel_m_read_readvariableop1
-savev2_adam_head_2_bias_m_read_readvariableop3
/savev2_adam_head_3_kernel_m_read_readvariableop1
-savev2_adam_head_3_bias_m_read_readvariableop3
/savev2_adam_head_4_kernel_m_read_readvariableop1
-savev2_adam_head_4_bias_m_read_readvariableop3
/savev2_adam_head_5_kernel_m_read_readvariableop1
-savev2_adam_head_5_bias_m_read_readvariableop3
/savev2_adam_head_6_kernel_m_read_readvariableop1
-savev2_adam_head_6_bias_m_read_readvariableop3
/savev2_adam_head_7_kernel_m_read_readvariableop1
-savev2_adam_head_7_bias_m_read_readvariableop3
/savev2_adam_conv_1_kernel_m_read_readvariableop1
-savev2_adam_conv_1_bias_m_read_readvariableop3
/savev2_adam_conv_2_kernel_m_read_readvariableop1
-savev2_adam_conv_2_bias_m_read_readvariableop3
/savev2_adam_conv_3_kernel_m_read_readvariableop1
-savev2_adam_conv_3_bias_m_read_readvariableop3
/savev2_adam_dens_1_kernel_v_read_readvariableop1
-savev2_adam_dens_1_bias_v_read_readvariableop3
/savev2_adam_dens_2_kernel_v_read_readvariableop1
-savev2_adam_dens_2_bias_v_read_readvariableop3
/savev2_adam_dens_3_kernel_v_read_readvariableop1
-savev2_adam_dens_3_bias_v_read_readvariableop3
/savev2_adam_dens_4_kernel_v_read_readvariableop1
-savev2_adam_dens_4_bias_v_read_readvariableop3
/savev2_adam_dens_5_kernel_v_read_readvariableop1
-savev2_adam_dens_5_bias_v_read_readvariableop3
/savev2_adam_dens_6_kernel_v_read_readvariableop1
-savev2_adam_dens_6_bias_v_read_readvariableop3
/savev2_adam_dens_7_kernel_v_read_readvariableop1
-savev2_adam_dens_7_bias_v_read_readvariableop3
/savev2_adam_head_1_kernel_v_read_readvariableop1
-savev2_adam_head_1_bias_v_read_readvariableop3
/savev2_adam_head_2_kernel_v_read_readvariableop1
-savev2_adam_head_2_bias_v_read_readvariableop3
/savev2_adam_head_3_kernel_v_read_readvariableop1
-savev2_adam_head_3_bias_v_read_readvariableop3
/savev2_adam_head_4_kernel_v_read_readvariableop1
-savev2_adam_head_4_bias_v_read_readvariableop3
/savev2_adam_head_5_kernel_v_read_readvariableop1
-savev2_adam_head_5_bias_v_read_readvariableop3
/savev2_adam_head_6_kernel_v_read_readvariableop1
-savev2_adam_head_6_bias_v_read_readvariableop3
/savev2_adam_head_7_kernel_v_read_readvariableop1
-savev2_adam_head_7_bias_v_read_readvariableop3
/savev2_adam_conv_1_kernel_v_read_readvariableop1
-savev2_adam_conv_1_bias_v_read_readvariableop3
/savev2_adam_conv_2_kernel_v_read_readvariableop1
-savev2_adam_conv_2_bias_v_read_readvariableop3
/savev2_adam_conv_3_kernel_v_read_readvariableop1
-savev2_adam_conv_3_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�H
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�G
value�GB�G�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�1
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dens_1_kernel_read_readvariableop&savev2_dens_1_bias_read_readvariableop(savev2_dens_2_kernel_read_readvariableop&savev2_dens_2_bias_read_readvariableop(savev2_dens_3_kernel_read_readvariableop&savev2_dens_3_bias_read_readvariableop(savev2_dens_4_kernel_read_readvariableop&savev2_dens_4_bias_read_readvariableop(savev2_dens_5_kernel_read_readvariableop&savev2_dens_5_bias_read_readvariableop(savev2_dens_6_kernel_read_readvariableop&savev2_dens_6_bias_read_readvariableop(savev2_dens_7_kernel_read_readvariableop&savev2_dens_7_bias_read_readvariableop(savev2_head_1_kernel_read_readvariableop&savev2_head_1_bias_read_readvariableop(savev2_head_2_kernel_read_readvariableop&savev2_head_2_bias_read_readvariableop(savev2_head_3_kernel_read_readvariableop&savev2_head_3_bias_read_readvariableop(savev2_head_4_kernel_read_readvariableop&savev2_head_4_bias_read_readvariableop(savev2_head_5_kernel_read_readvariableop&savev2_head_5_bias_read_readvariableop(savev2_head_6_kernel_read_readvariableop&savev2_head_6_bias_read_readvariableop(savev2_head_7_kernel_read_readvariableop&savev2_head_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_7_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_8_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_9_read_readvariableop"savev2_count_9_read_readvariableop#savev2_total_10_read_readvariableop#savev2_count_10_read_readvariableop#savev2_total_11_read_readvariableop#savev2_count_11_read_readvariableop#savev2_total_12_read_readvariableop#savev2_count_12_read_readvariableop#savev2_total_13_read_readvariableop#savev2_count_13_read_readvariableop#savev2_total_14_read_readvariableop#savev2_count_14_read_readvariableop/savev2_adam_dens_1_kernel_m_read_readvariableop-savev2_adam_dens_1_bias_m_read_readvariableop/savev2_adam_dens_2_kernel_m_read_readvariableop-savev2_adam_dens_2_bias_m_read_readvariableop/savev2_adam_dens_3_kernel_m_read_readvariableop-savev2_adam_dens_3_bias_m_read_readvariableop/savev2_adam_dens_4_kernel_m_read_readvariableop-savev2_adam_dens_4_bias_m_read_readvariableop/savev2_adam_dens_5_kernel_m_read_readvariableop-savev2_adam_dens_5_bias_m_read_readvariableop/savev2_adam_dens_6_kernel_m_read_readvariableop-savev2_adam_dens_6_bias_m_read_readvariableop/savev2_adam_dens_7_kernel_m_read_readvariableop-savev2_adam_dens_7_bias_m_read_readvariableop/savev2_adam_head_1_kernel_m_read_readvariableop-savev2_adam_head_1_bias_m_read_readvariableop/savev2_adam_head_2_kernel_m_read_readvariableop-savev2_adam_head_2_bias_m_read_readvariableop/savev2_adam_head_3_kernel_m_read_readvariableop-savev2_adam_head_3_bias_m_read_readvariableop/savev2_adam_head_4_kernel_m_read_readvariableop-savev2_adam_head_4_bias_m_read_readvariableop/savev2_adam_head_5_kernel_m_read_readvariableop-savev2_adam_head_5_bias_m_read_readvariableop/savev2_adam_head_6_kernel_m_read_readvariableop-savev2_adam_head_6_bias_m_read_readvariableop/savev2_adam_head_7_kernel_m_read_readvariableop-savev2_adam_head_7_bias_m_read_readvariableop/savev2_adam_conv_1_kernel_m_read_readvariableop-savev2_adam_conv_1_bias_m_read_readvariableop/savev2_adam_conv_2_kernel_m_read_readvariableop-savev2_adam_conv_2_bias_m_read_readvariableop/savev2_adam_conv_3_kernel_m_read_readvariableop-savev2_adam_conv_3_bias_m_read_readvariableop/savev2_adam_dens_1_kernel_v_read_readvariableop-savev2_adam_dens_1_bias_v_read_readvariableop/savev2_adam_dens_2_kernel_v_read_readvariableop-savev2_adam_dens_2_bias_v_read_readvariableop/savev2_adam_dens_3_kernel_v_read_readvariableop-savev2_adam_dens_3_bias_v_read_readvariableop/savev2_adam_dens_4_kernel_v_read_readvariableop-savev2_adam_dens_4_bias_v_read_readvariableop/savev2_adam_dens_5_kernel_v_read_readvariableop-savev2_adam_dens_5_bias_v_read_readvariableop/savev2_adam_dens_6_kernel_v_read_readvariableop-savev2_adam_dens_6_bias_v_read_readvariableop/savev2_adam_dens_7_kernel_v_read_readvariableop-savev2_adam_dens_7_bias_v_read_readvariableop/savev2_adam_head_1_kernel_v_read_readvariableop-savev2_adam_head_1_bias_v_read_readvariableop/savev2_adam_head_2_kernel_v_read_readvariableop-savev2_adam_head_2_bias_v_read_readvariableop/savev2_adam_head_3_kernel_v_read_readvariableop-savev2_adam_head_3_bias_v_read_readvariableop/savev2_adam_head_4_kernel_v_read_readvariableop-savev2_adam_head_4_bias_v_read_readvariableop/savev2_adam_head_5_kernel_v_read_readvariableop-savev2_adam_head_5_bias_v_read_readvariableop/savev2_adam_head_6_kernel_v_read_readvariableop-savev2_adam_head_6_bias_v_read_readvariableop/savev2_adam_head_7_kernel_v_read_readvariableop-savev2_adam_head_7_bias_v_read_readvariableop/savev2_adam_conv_1_kernel_v_read_readvariableop-savev2_adam_conv_1_bias_v_read_readvariableop/savev2_adam_conv_2_kernel_v_read_readvariableop-savev2_adam_conv_2_bias_v_read_readvariableop/savev2_adam_conv_3_kernel_v_read_readvariableop-savev2_adam_conv_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
�2�	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:
��:�:
��:�:
��:�:
��:�:
��:�:
��:�:	�::	�::	�::	�::	�::	�::	�:: : : : : : : : @:@:@`:`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
��:�:
��:�:
��:�:
��:�:
��:�:
��:�:
��:�:	�::	�::	�::	�::	�::	�::	�:: : : @:@:@`:`:
��:�:
��:�:
��:�:
��:�:
��:�:
��:�:
��:�:	�::	�::	�::	�::	�::	�::	�:: : : @:@:@`:`: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&	"
 
_output_shapes
:
��:!


_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :("$
"
_output_shapes
: : #

_output_shapes
: :($$
"
_output_shapes
: @: %

_output_shapes
:@:(&$
"
_output_shapes
:@`: '

_output_shapes
:`:(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :&F"
 
_output_shapes
:
��:!G

_output_shapes	
:�:&H"
 
_output_shapes
:
��:!I

_output_shapes	
:�:&J"
 
_output_shapes
:
��:!K

_output_shapes	
:�:&L"
 
_output_shapes
:
��:!M

_output_shapes	
:�:&N"
 
_output_shapes
:
��:!O

_output_shapes	
:�:&P"
 
_output_shapes
:
��:!Q

_output_shapes	
:�:&R"
 
_output_shapes
:
��:!S

_output_shapes	
:�:%T!

_output_shapes
:	�: U

_output_shapes
::%V!

_output_shapes
:	�: W

_output_shapes
::%X!

_output_shapes
:	�: Y

_output_shapes
::%Z!

_output_shapes
:	�: [

_output_shapes
::%\!

_output_shapes
:	�: ]

_output_shapes
::%^!

_output_shapes
:	�: _

_output_shapes
::%`!

_output_shapes
:	�: a

_output_shapes
::(b$
"
_output_shapes
: : c

_output_shapes
: :(d$
"
_output_shapes
: @: e

_output_shapes
:@:(f$
"
_output_shapes
:@`: g

_output_shapes
:`:&h"
 
_output_shapes
:
��:!i

_output_shapes	
:�:&j"
 
_output_shapes
:
��:!k

_output_shapes	
:�:&l"
 
_output_shapes
:
��:!m

_output_shapes	
:�:&n"
 
_output_shapes
:
��:!o

_output_shapes	
:�:&p"
 
_output_shapes
:
��:!q

_output_shapes	
:�:&r"
 
_output_shapes
:
��:!s

_output_shapes	
:�:&t"
 
_output_shapes
:
��:!u

_output_shapes	
:�:%v!

_output_shapes
:	�: w

_output_shapes
::%x!

_output_shapes
:	�: y

_output_shapes
::%z!

_output_shapes
:	�: {

_output_shapes
::%|!

_output_shapes
:	�: }

_output_shapes
::%~!

_output_shapes
:	�: 

_output_shapes
::&�!

_output_shapes
:	�:!�

_output_shapes
::&�!

_output_shapes
:	�:!�

_output_shapes
::)�$
"
_output_shapes
: :!�

_output_shapes
: :)�$
"
_output_shapes
: @:!�

_output_shapes
:@:)�$
"
_output_shapes
:@`:!�

_output_shapes
:`:�

_output_shapes
: 
�
�
B__inference_conv_1_layer_call_and_return_conditional_losses_216434

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�/conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������i 2
Relu�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp0^conv_1/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������i 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�<
�
B__inference_trunk__layer_call_and_return_conditional_losses_216709

inputs#
conv_1_216672: 
conv_1_216674: #
conv_2_216678: @
conv_2_216680:@#
conv_3_216684:@`
conv_3_216686:`
identity��conv_1/StatefulPartitionedCall�/conv_1/kernel/Regularizer/Square/ReadVariableOp�conv_2/StatefulPartitionedCall�/conv_2/kernel/Regularizer/Square/ReadVariableOp�conv_3/StatefulPartitionedCall�/conv_3/kernel/Regularizer/Square/ReadVariableOp�drop_1/StatefulPartitionedCall�drop_2/StatefulPartitionedCall�drop_3/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_216672conv_1_216674*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_2164342 
conv_1/StatefulPartitionedCall�
drop_1/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_1_layer_call_and_return_conditional_losses_2166372 
drop_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'drop_1/StatefulPartitionedCall:output:0conv_2_216678conv_2_216680*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_2164692 
conv_2/StatefulPartitionedCall�
drop_2/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0^drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_2_layer_call_and_return_conditional_losses_2166042 
drop_2/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall'drop_2/StatefulPartitionedCall:output:0conv_3_216684conv_3_216686*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_2165042 
conv_3/StatefulPartitionedCall�
drop_3/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0^drop_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_2165712 
drop_3/StatefulPartitionedCall�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_1_216672*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_2_216678*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_3_216684*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentity'drop_3/StatefulPartitionedCall:output:0^conv_1/StatefulPartitionedCall0^conv_1/kernel/Regularizer/Square/ReadVariableOp^conv_2/StatefulPartitionedCall0^conv_2/kernel/Regularizer/Square/ReadVariableOp^conv_3/StatefulPartitionedCall0^conv_3/kernel/Regularizer/Square/ReadVariableOp^drop_1/StatefulPartitionedCall^drop_2/StatefulPartitionedCall^drop_3/StatefulPartitionedCall*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp2@
drop_1/StatefulPartitionedCalldrop_1/StatefulPartitionedCall2@
drop_2/StatefulPartitionedCalldrop_2/StatefulPartitionedCall2@
drop_3/StatefulPartitionedCalldrop_3/StatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
M
1__inference_global_max_pool__layer_call_fn_219701

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168532
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�
h
L__inference_global_max_pool__layer_call_and_return_conditional_losses_216853

inputs
identity�
pool_/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������)`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_pool__layer_call_and_return_conditional_losses_2168302
pool_/PartitionedCall�
flat_/PartitionedCallPartitionedCallpool_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_flat__layer_call_and_return_conditional_losses_2168502
flat_/PartitionedCalls
IdentityIdentityflat_/PartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�
M
1__inference_global_max_pool__layer_call_fn_219706

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_global_max_pool__layer_call_and_return_conditional_losses_2168752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�q
�
B__inference_trunk__layer_call_and_return_conditional_losses_219696

inputsH
2conv_1_conv1d_expanddims_1_readvariableop_resource: 4
&conv_1_biasadd_readvariableop_resource: H
2conv_2_conv1d_expanddims_1_readvariableop_resource: @4
&conv_2_biasadd_readvariableop_resource:@H
2conv_3_conv1d_expanddims_1_readvariableop_resource:@`4
&conv_3_biasadd_readvariableop_resource:`
identity��conv_1/BiasAdd/ReadVariableOp�)conv_1/conv1d/ExpandDims_1/ReadVariableOp�/conv_1/kernel/Regularizer/Square/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�)conv_2/conv1d/ExpandDims_1/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�)conv_3/conv1d/ExpandDims_1/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOp�
conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_1/conv1d/ExpandDims/dim�
conv_1/conv1d/ExpandDims
ExpandDimsinputs%conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv_1/conv1d/ExpandDims�
)conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02+
)conv_1/conv1d/ExpandDims_1/ReadVariableOp�
conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_1/conv1d/ExpandDims_1/dim�
conv_1/conv1d/ExpandDims_1
ExpandDims1conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv_1/conv1d/ExpandDims_1�
conv_1/conv1dConv2D!conv_1/conv1d/ExpandDims:output:0#conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
conv_1/conv1d�
conv_1/conv1d/SqueezeSqueezeconv_1/conv1d:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2
conv_1/conv1d/Squeeze�
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_1/BiasAdd/ReadVariableOp�
conv_1/BiasAddBiasAddconv_1/conv1d/Squeeze:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
conv_1/BiasAddq
conv_1/ReluReluconv_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������i 2
conv_1/Reluq
drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
drop_1/dropout/Const�
drop_1/dropout/MulMulconv_1/Relu:activations:0drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:���������i 2
drop_1/dropout/Mulu
drop_1/dropout/ShapeShapeconv_1/Relu:activations:0*
T0*
_output_shapes
:2
drop_1/dropout/Shape�
+drop_1/dropout/random_uniform/RandomUniformRandomUniformdrop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype02-
+drop_1/dropout/random_uniform/RandomUniform�
drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
drop_1/dropout/GreaterEqual/y�
drop_1/dropout/GreaterEqualGreaterEqual4drop_1/dropout/random_uniform/RandomUniform:output:0&drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2
drop_1/dropout/GreaterEqual�
drop_1/dropout/CastCastdrop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
drop_1/dropout/Cast�
drop_1/dropout/Mul_1Muldrop_1/dropout/Mul:z:0drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:���������i 2
drop_1/dropout/Mul_1�
conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_2/conv1d/ExpandDims/dim�
conv_2/conv1d/ExpandDims
ExpandDimsdrop_1/dropout/Mul_1:z:0%conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2
conv_2/conv1d/ExpandDims�
)conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02+
)conv_2/conv1d/ExpandDims_1/ReadVariableOp�
conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_2/conv1d/ExpandDims_1/dim�
conv_2/conv1d/ExpandDims_1
ExpandDims1conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv_2/conv1d/ExpandDims_1�
conv_2/conv1dConv2D!conv_2/conv1d/ExpandDims:output:0#conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
conv_2/conv1d�
conv_2/conv1d/SqueezeSqueezeconv_2/conv1d:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2
conv_2/conv1d/Squeeze�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_2/BiasAdd/ReadVariableOp�
conv_2/BiasAddBiasAddconv_2/conv1d/Squeeze:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
conv_2/BiasAddq
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������Z@2
conv_2/Reluq
drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
drop_2/dropout/Const�
drop_2/dropout/MulMulconv_2/Relu:activations:0drop_2/dropout/Const:output:0*
T0*+
_output_shapes
:���������Z@2
drop_2/dropout/Mulu
drop_2/dropout/ShapeShapeconv_2/Relu:activations:0*
T0*
_output_shapes
:2
drop_2/dropout/Shape�
+drop_2/dropout/random_uniform/RandomUniformRandomUniformdrop_2/dropout/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype02-
+drop_2/dropout/random_uniform/RandomUniform�
drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
drop_2/dropout/GreaterEqual/y�
drop_2/dropout/GreaterEqualGreaterEqual4drop_2/dropout/random_uniform/RandomUniform:output:0&drop_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2
drop_2/dropout/GreaterEqual�
drop_2/dropout/CastCastdrop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
drop_2/dropout/Cast�
drop_2/dropout/Mul_1Muldrop_2/dropout/Mul:z:0drop_2/dropout/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
drop_2/dropout/Mul_1�
conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_3/conv1d/ExpandDims/dim�
conv_3/conv1d/ExpandDims
ExpandDimsdrop_2/dropout/Mul_1:z:0%conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2
conv_3/conv1d/ExpandDims�
)conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02+
)conv_3/conv1d/ExpandDims_1/ReadVariableOp�
conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_3/conv1d/ExpandDims_1/dim�
conv_3/conv1d/ExpandDims_1
ExpandDims1conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2
conv_3/conv1d/ExpandDims_1�
conv_3/conv1dConv2D!conv_3/conv1d/ExpandDims:output:0#conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
conv_3/conv1d�
conv_3/conv1d/SqueezeSqueezeconv_3/conv1d:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2
conv_3/conv1d/Squeeze�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/conv1d/Squeeze:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
conv_3/BiasAddq
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������S`2
conv_3/Reluq
drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
drop_3/dropout/Const�
drop_3/dropout/MulMulconv_3/Relu:activations:0drop_3/dropout/Const:output:0*
T0*+
_output_shapes
:���������S`2
drop_3/dropout/Mulu
drop_3/dropout/ShapeShapeconv_3/Relu:activations:0*
T0*
_output_shapes
:2
drop_3/dropout/Shape�
+drop_3/dropout/random_uniform/RandomUniformRandomUniformdrop_3/dropout/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype02-
+drop_3/dropout/random_uniform/RandomUniform�
drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
drop_3/dropout/GreaterEqual/y�
drop_3/dropout/GreaterEqualGreaterEqual4drop_3/dropout/random_uniform/RandomUniform:output:0&drop_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2
drop_3/dropout/GreaterEqual�
drop_3/dropout/CastCastdrop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
drop_3/dropout/Cast�
drop_3/dropout/Mul_1Muldrop_3/dropout/Mul:z:0drop_3/dropout/Cast:y:0*
T0*+
_output_shapes
:���������S`2
drop_3/dropout/Mul_1�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentitydrop_3/dropout/Mul_1:z:0^conv_1/BiasAdd/ReadVariableOp*^conv_1/conv1d/ExpandDims_1/ReadVariableOp0^conv_1/kernel/Regularizer/Square/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp*^conv_2/conv1d/ExpandDims_1/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp*^conv_3/conv1d/ExpandDims_1/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2V
)conv_1/conv1d/ExpandDims_1/ReadVariableOp)conv_1/conv1d/ExpandDims_1/ReadVariableOp2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2V
)conv_2/conv1d/ExpandDims_1/ReadVariableOp)conv_2/conv1d/ExpandDims_1/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2V
)conv_3/conv1d/ExpandDims_1/ReadVariableOp)conv_3/conv1d/ExpandDims_1/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_head_5_layer_call_fn_219955

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_head_5_layer_call_and_return_conditional_losses_2171402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
B__inference_drop_3_layer_call_and_return_conditional_losses_216571

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������S`2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������S`2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�
�
'__inference_trunk__layer_call_fn_216551

input_
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@`
	unknown_4:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinput_
�7
�
B__inference_trunk__layer_call_and_return_conditional_losses_216781

input_#
conv_1_216744: 
conv_1_216746: #
conv_2_216750: @
conv_2_216752:@#
conv_3_216756:@`
conv_3_216758:`
identity��conv_1/StatefulPartitionedCall�/conv_1/kernel/Regularizer/Square/ReadVariableOp�conv_2/StatefulPartitionedCall�/conv_2/kernel/Regularizer/Square/ReadVariableOp�conv_3/StatefulPartitionedCall�/conv_3/kernel/Regularizer/Square/ReadVariableOp�
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_conv_1_216744conv_1_216746*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_2164342 
conv_1/StatefulPartitionedCall�
drop_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_1_layer_call_and_return_conditional_losses_2164452
drop_1/PartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCalldrop_1/PartitionedCall:output:0conv_2_216750conv_2_216752*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_2164692 
conv_2/StatefulPartitionedCall�
drop_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_2_layer_call_and_return_conditional_losses_2164802
drop_2/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCalldrop_2/PartitionedCall:output:0conv_3_216756conv_3_216758*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_2165042 
conv_3/StatefulPartitionedCall�
drop_3/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_2165152
drop_3/PartitionedCall�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_1_216744*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_2_216750*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_3_216756*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentitydrop_3/PartitionedCall:output:0^conv_1/StatefulPartitionedCall0^conv_1/kernel/Regularizer/Square/ReadVariableOp^conv_2/StatefulPartitionedCall0^conv_2/kernel/Regularizer/Square/ReadVariableOp^conv_3/StatefulPartitionedCall0^conv_3/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinput_
�
�
'__inference_dens_1_layer_call_fn_219735

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_1_layer_call_and_return_conditional_losses_2170892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_head_5_layer_call_and_return_conditional_losses_219966

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_220209N
8conv_1_kernel_regularizer_square_readvariableop_resource: 
identity��/conv_1/kernel/Regularizer/Square/ReadVariableOp�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv_1_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
IdentityIdentity!conv_1/kernel/Regularizer/mul:z:00^conv_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp
�
`
B__inference_drop_1_layer_call_and_return_conditional_losses_216445

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������i 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������i 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������i :S O
+
_output_shapes
:���������i 
 
_user_specified_nameinputs
�
�
'__inference_trunk__layer_call_fn_216741

input_
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@`
	unknown_4:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2167092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinput_
�
�
'__inference_conv_2_layer_call_fn_220085

inputs
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_2164692
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������i : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������i 
 
_user_specified_nameinputs
�
�
B__inference_conv_2_layer_call_and_return_conditional_losses_220107

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������Z@2
Relu�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������i : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������i 
 
_user_specified_nameinputs
�
a
B__inference_drop_2_layer_call_and_return_conditional_losses_220134

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:���������Z@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2
dropout/GreaterEqual�
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:���������Z@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z@:S O
+
_output_shapes
:���������Z@
 
_user_specified_nameinputs
�
]
A__inference_flat__layer_call_and_return_conditional_losses_220242

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������)`:S O
+
_output_shapes
:���������)`
 
_user_specified_nameinputs
�
`
B__inference_drop_2_layer_call_and_return_conditional_losses_216480

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������Z@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������Z@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Z@:S O
+
_output_shapes
:���������Z@
 
_user_specified_nameinputs
��
�)
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_219501
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6O
9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource: ;
-trunk__conv_1_biasadd_readvariableop_resource: O
9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource: @;
-trunk__conv_2_biasadd_readvariableop_resource:@O
9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource:@`;
-trunk__conv_3_biasadd_readvariableop_resource:`9
%dens_7_matmul_readvariableop_resource:
��5
&dens_7_biasadd_readvariableop_resource:	�9
%dens_6_matmul_readvariableop_resource:
��5
&dens_6_biasadd_readvariableop_resource:	�9
%dens_5_matmul_readvariableop_resource:
��5
&dens_5_biasadd_readvariableop_resource:	�9
%dens_4_matmul_readvariableop_resource:
��5
&dens_4_biasadd_readvariableop_resource:	�9
%dens_3_matmul_readvariableop_resource:
��5
&dens_3_biasadd_readvariableop_resource:	�9
%dens_2_matmul_readvariableop_resource:
��5
&dens_2_biasadd_readvariableop_resource:	�9
%dens_1_matmul_readvariableop_resource:
��5
&dens_1_biasadd_readvariableop_resource:	�8
%head_7_matmul_readvariableop_resource:	�4
&head_7_biasadd_readvariableop_resource:8
%head_6_matmul_readvariableop_resource:	�4
&head_6_biasadd_readvariableop_resource:8
%head_5_matmul_readvariableop_resource:	�4
&head_5_biasadd_readvariableop_resource:8
%head_4_matmul_readvariableop_resource:	�4
&head_4_biasadd_readvariableop_resource:8
%head_3_matmul_readvariableop_resource:	�4
&head_3_biasadd_readvariableop_resource:8
%head_2_matmul_readvariableop_resource:	�4
&head_2_biasadd_readvariableop_resource:8
%head_1_matmul_readvariableop_resource:	�4
&head_1_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��/conv_1/kernel/Regularizer/Square/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOp�dens_1/BiasAdd/ReadVariableOp�dens_1/MatMul/ReadVariableOp�dens_2/BiasAdd/ReadVariableOp�dens_2/MatMul/ReadVariableOp�dens_3/BiasAdd/ReadVariableOp�dens_3/MatMul/ReadVariableOp�dens_4/BiasAdd/ReadVariableOp�dens_4/MatMul/ReadVariableOp�dens_5/BiasAdd/ReadVariableOp�dens_5/MatMul/ReadVariableOp�dens_6/BiasAdd/ReadVariableOp�dens_6/MatMul/ReadVariableOp�dens_7/BiasAdd/ReadVariableOp�dens_7/MatMul/ReadVariableOp�head_1/BiasAdd/ReadVariableOp�head_1/MatMul/ReadVariableOp�head_2/BiasAdd/ReadVariableOp�head_2/MatMul/ReadVariableOp�head_3/BiasAdd/ReadVariableOp�head_3/MatMul/ReadVariableOp�head_4/BiasAdd/ReadVariableOp�head_4/MatMul/ReadVariableOp�head_5/BiasAdd/ReadVariableOp�head_5/MatMul/ReadVariableOp�head_6/BiasAdd/ReadVariableOp�head_6/MatMul/ReadVariableOp�head_7/BiasAdd/ReadVariableOp�head_7/MatMul/ReadVariableOp�$trunk_/conv_1/BiasAdd/ReadVariableOp�&trunk_/conv_1/BiasAdd_1/ReadVariableOp�&trunk_/conv_1/BiasAdd_2/ReadVariableOp�&trunk_/conv_1/BiasAdd_3/ReadVariableOp�&trunk_/conv_1/BiasAdd_4/ReadVariableOp�&trunk_/conv_1/BiasAdd_5/ReadVariableOp�&trunk_/conv_1/BiasAdd_6/ReadVariableOp�0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp�$trunk_/conv_2/BiasAdd/ReadVariableOp�&trunk_/conv_2/BiasAdd_1/ReadVariableOp�&trunk_/conv_2/BiasAdd_2/ReadVariableOp�&trunk_/conv_2/BiasAdd_3/ReadVariableOp�&trunk_/conv_2/BiasAdd_4/ReadVariableOp�&trunk_/conv_2/BiasAdd_5/ReadVariableOp�&trunk_/conv_2/BiasAdd_6/ReadVariableOp�0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp�$trunk_/conv_3/BiasAdd/ReadVariableOp�&trunk_/conv_3/BiasAdd_1/ReadVariableOp�&trunk_/conv_3/BiasAdd_2/ReadVariableOp�&trunk_/conv_3/BiasAdd_3/ReadVariableOp�&trunk_/conv_3/BiasAdd_4/ReadVariableOp�&trunk_/conv_3/BiasAdd_5/ReadVariableOp�&trunk_/conv_3/BiasAdd_6/ReadVariableOp�0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp�
#trunk_/conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#trunk_/conv_1/conv1d/ExpandDims/dim�
trunk_/conv_1/conv1d/ExpandDims
ExpandDimsinputs_6,trunk_/conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2!
trunk_/conv_1/conv1d/ExpandDims�
0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype022
0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp�
%trunk_/conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%trunk_/conv_1/conv1d/ExpandDims_1/dim�
!trunk_/conv_1/conv1d/ExpandDims_1
ExpandDims8trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0.trunk_/conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2#
!trunk_/conv_1/conv1d/ExpandDims_1�
trunk_/conv_1/conv1dConv2D(trunk_/conv_1/conv1d/ExpandDims:output:0*trunk_/conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d�
trunk_/conv_1/conv1d/SqueezeSqueezetrunk_/conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2
trunk_/conv_1/conv1d/Squeeze�
$trunk_/conv_1/BiasAdd/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$trunk_/conv_1/BiasAdd/ReadVariableOp�
trunk_/conv_1/BiasAddBiasAdd%trunk_/conv_1/conv1d/Squeeze:output:0,trunk_/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd�
trunk_/conv_1/ReluRelutrunk_/conv_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu
trunk_/drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_1/dropout/Const�
trunk_/drop_1/dropout/MulMul trunk_/conv_1/Relu:activations:0$trunk_/drop_1/dropout/Const:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout/Mul�
trunk_/drop_1/dropout/ShapeShape trunk_/conv_1/Relu:activations:0*
T0*
_output_shapes
:2
trunk_/drop_1/dropout/Shape�
2trunk_/drop_1/dropout/random_uniform/RandomUniformRandomUniform$trunk_/drop_1/dropout/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype024
2trunk_/drop_1/dropout/random_uniform/RandomUniform�
$trunk_/drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2&
$trunk_/drop_1/dropout/GreaterEqual/y�
"trunk_/drop_1/dropout/GreaterEqualGreaterEqual;trunk_/drop_1/dropout/random_uniform/RandomUniform:output:0-trunk_/drop_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2$
"trunk_/drop_1/dropout/GreaterEqual�
trunk_/drop_1/dropout/CastCast&trunk_/drop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout/Cast�
trunk_/drop_1/dropout/Mul_1Multrunk_/drop_1/dropout/Mul:z:0trunk_/drop_1/dropout/Cast:y:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout/Mul_1�
#trunk_/conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#trunk_/conv_2/conv1d/ExpandDims/dim�
trunk_/conv_2/conv1d/ExpandDims
ExpandDimstrunk_/drop_1/dropout/Mul_1:z:0,trunk_/conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2!
trunk_/conv_2/conv1d/ExpandDims�
0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype022
0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp�
%trunk_/conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%trunk_/conv_2/conv1d/ExpandDims_1/dim�
!trunk_/conv_2/conv1d/ExpandDims_1
ExpandDims8trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0.trunk_/conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2#
!trunk_/conv_2/conv1d/ExpandDims_1�
trunk_/conv_2/conv1dConv2D(trunk_/conv_2/conv1d/ExpandDims:output:0*trunk_/conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d�
trunk_/conv_2/conv1d/SqueezeSqueezetrunk_/conv_2/conv1d:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2
trunk_/conv_2/conv1d/Squeeze�
$trunk_/conv_2/BiasAdd/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$trunk_/conv_2/BiasAdd/ReadVariableOp�
trunk_/conv_2/BiasAddBiasAdd%trunk_/conv_2/conv1d/Squeeze:output:0,trunk_/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd�
trunk_/conv_2/ReluRelutrunk_/conv_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu
trunk_/drop_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_2/dropout/Const�
trunk_/drop_2/dropout/MulMul trunk_/conv_2/Relu:activations:0$trunk_/drop_2/dropout/Const:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout/Mul�
trunk_/drop_2/dropout/ShapeShape trunk_/conv_2/Relu:activations:0*
T0*
_output_shapes
:2
trunk_/drop_2/dropout/Shape�
2trunk_/drop_2/dropout/random_uniform/RandomUniformRandomUniform$trunk_/drop_2/dropout/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype024
2trunk_/drop_2/dropout/random_uniform/RandomUniform�
$trunk_/drop_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2&
$trunk_/drop_2/dropout/GreaterEqual/y�
"trunk_/drop_2/dropout/GreaterEqualGreaterEqual;trunk_/drop_2/dropout/random_uniform/RandomUniform:output:0-trunk_/drop_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2$
"trunk_/drop_2/dropout/GreaterEqual�
trunk_/drop_2/dropout/CastCast&trunk_/drop_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout/Cast�
trunk_/drop_2/dropout/Mul_1Multrunk_/drop_2/dropout/Mul:z:0trunk_/drop_2/dropout/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout/Mul_1�
#trunk_/conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#trunk_/conv_3/conv1d/ExpandDims/dim�
trunk_/conv_3/conv1d/ExpandDims
ExpandDimstrunk_/drop_2/dropout/Mul_1:z:0,trunk_/conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2!
trunk_/conv_3/conv1d/ExpandDims�
0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype022
0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp�
%trunk_/conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%trunk_/conv_3/conv1d/ExpandDims_1/dim�
!trunk_/conv_3/conv1d/ExpandDims_1
ExpandDims8trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0.trunk_/conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2#
!trunk_/conv_3/conv1d/ExpandDims_1�
trunk_/conv_3/conv1dConv2D(trunk_/conv_3/conv1d/ExpandDims:output:0*trunk_/conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d�
trunk_/conv_3/conv1d/SqueezeSqueezetrunk_/conv_3/conv1d:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2
trunk_/conv_3/conv1d/Squeeze�
$trunk_/conv_3/BiasAdd/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02&
$trunk_/conv_3/BiasAdd/ReadVariableOp�
trunk_/conv_3/BiasAddBiasAdd%trunk_/conv_3/conv1d/Squeeze:output:0,trunk_/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd�
trunk_/conv_3/ReluRelutrunk_/conv_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu
trunk_/drop_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_3/dropout/Const�
trunk_/drop_3/dropout/MulMul trunk_/conv_3/Relu:activations:0$trunk_/drop_3/dropout/Const:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout/Mul�
trunk_/drop_3/dropout/ShapeShape trunk_/conv_3/Relu:activations:0*
T0*
_output_shapes
:2
trunk_/drop_3/dropout/Shape�
2trunk_/drop_3/dropout/random_uniform/RandomUniformRandomUniform$trunk_/drop_3/dropout/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype024
2trunk_/drop_3/dropout/random_uniform/RandomUniform�
$trunk_/drop_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2&
$trunk_/drop_3/dropout/GreaterEqual/y�
"trunk_/drop_3/dropout/GreaterEqualGreaterEqual;trunk_/drop_3/dropout/random_uniform/RandomUniform:output:0-trunk_/drop_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2$
"trunk_/drop_3/dropout/GreaterEqual�
trunk_/drop_3/dropout/CastCast&trunk_/drop_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout/Cast�
trunk_/drop_3/dropout/Mul_1Multrunk_/drop_3/dropout/Mul:z:0trunk_/drop_3/dropout/Cast:y:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout/Mul_1�
%trunk_/conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_1/ExpandDims/dim�
!trunk_/conv_1/conv1d_1/ExpandDims
ExpandDimsinputs_5.trunk_/conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_1/ExpandDims�
2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_1/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_1/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_1/ExpandDims_1�
trunk_/conv_1/conv1d_1Conv2D*trunk_/conv_1/conv1d_1/ExpandDims:output:0,trunk_/conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_1�
trunk_/conv_1/conv1d_1/SqueezeSqueezetrunk_/conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_1/Squeeze�
&trunk_/conv_1/BiasAdd_1/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_1/ReadVariableOp�
trunk_/conv_1/BiasAdd_1BiasAdd'trunk_/conv_1/conv1d_1/Squeeze:output:0.trunk_/conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_1�
trunk_/conv_1/Relu_1Relu trunk_/conv_1/BiasAdd_1:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_1�
trunk_/drop_1/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_1/dropout_1/Const�
trunk_/drop_1/dropout_1/MulMul"trunk_/conv_1/Relu_1:activations:0&trunk_/drop_1/dropout_1/Const:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_1/Mul�
trunk_/drop_1/dropout_1/ShapeShape"trunk_/conv_1/Relu_1:activations:0*
T0*
_output_shapes
:2
trunk_/drop_1/dropout_1/Shape�
4trunk_/drop_1/dropout_1/random_uniform/RandomUniformRandomUniform&trunk_/drop_1/dropout_1/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype026
4trunk_/drop_1/dropout_1/random_uniform/RandomUniform�
&trunk_/drop_1/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_1/dropout_1/GreaterEqual/y�
$trunk_/drop_1/dropout_1/GreaterEqualGreaterEqual=trunk_/drop_1/dropout_1/random_uniform/RandomUniform:output:0/trunk_/drop_1/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2&
$trunk_/drop_1/dropout_1/GreaterEqual�
trunk_/drop_1/dropout_1/CastCast(trunk_/drop_1/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_1/Cast�
trunk_/drop_1/dropout_1/Mul_1Multrunk_/drop_1/dropout_1/Mul:z:0 trunk_/drop_1/dropout_1/Cast:y:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_1/Mul_1�
%trunk_/conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_1/ExpandDims/dim�
!trunk_/conv_2/conv1d_1/ExpandDims
ExpandDims!trunk_/drop_1/dropout_1/Mul_1:z:0.trunk_/conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_1/ExpandDims�
2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_1/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_1/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_1/ExpandDims_1�
trunk_/conv_2/conv1d_1Conv2D*trunk_/conv_2/conv1d_1/ExpandDims:output:0,trunk_/conv_2/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_1�
trunk_/conv_2/conv1d_1/SqueezeSqueezetrunk_/conv_2/conv1d_1:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_1/Squeeze�
&trunk_/conv_2/BiasAdd_1/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_1/ReadVariableOp�
trunk_/conv_2/BiasAdd_1BiasAdd'trunk_/conv_2/conv1d_1/Squeeze:output:0.trunk_/conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_1�
trunk_/conv_2/Relu_1Relu trunk_/conv_2/BiasAdd_1:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_1�
trunk_/drop_2/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_2/dropout_1/Const�
trunk_/drop_2/dropout_1/MulMul"trunk_/conv_2/Relu_1:activations:0&trunk_/drop_2/dropout_1/Const:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_1/Mul�
trunk_/drop_2/dropout_1/ShapeShape"trunk_/conv_2/Relu_1:activations:0*
T0*
_output_shapes
:2
trunk_/drop_2/dropout_1/Shape�
4trunk_/drop_2/dropout_1/random_uniform/RandomUniformRandomUniform&trunk_/drop_2/dropout_1/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype026
4trunk_/drop_2/dropout_1/random_uniform/RandomUniform�
&trunk_/drop_2/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_2/dropout_1/GreaterEqual/y�
$trunk_/drop_2/dropout_1/GreaterEqualGreaterEqual=trunk_/drop_2/dropout_1/random_uniform/RandomUniform:output:0/trunk_/drop_2/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2&
$trunk_/drop_2/dropout_1/GreaterEqual�
trunk_/drop_2/dropout_1/CastCast(trunk_/drop_2/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_1/Cast�
trunk_/drop_2/dropout_1/Mul_1Multrunk_/drop_2/dropout_1/Mul:z:0 trunk_/drop_2/dropout_1/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_1/Mul_1�
%trunk_/conv_3/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_1/ExpandDims/dim�
!trunk_/conv_3/conv1d_1/ExpandDims
ExpandDims!trunk_/drop_2/dropout_1/Mul_1:z:0.trunk_/conv_3/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_1/ExpandDims�
2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_1/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_1/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_1/ExpandDims_1�
trunk_/conv_3/conv1d_1Conv2D*trunk_/conv_3/conv1d_1/ExpandDims:output:0,trunk_/conv_3/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_1�
trunk_/conv_3/conv1d_1/SqueezeSqueezetrunk_/conv_3/conv1d_1:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_1/Squeeze�
&trunk_/conv_3/BiasAdd_1/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_1/ReadVariableOp�
trunk_/conv_3/BiasAdd_1BiasAdd'trunk_/conv_3/conv1d_1/Squeeze:output:0.trunk_/conv_3/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_1�
trunk_/conv_3/Relu_1Relu trunk_/conv_3/BiasAdd_1:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_1�
trunk_/drop_3/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_3/dropout_1/Const�
trunk_/drop_3/dropout_1/MulMul"trunk_/conv_3/Relu_1:activations:0&trunk_/drop_3/dropout_1/Const:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_1/Mul�
trunk_/drop_3/dropout_1/ShapeShape"trunk_/conv_3/Relu_1:activations:0*
T0*
_output_shapes
:2
trunk_/drop_3/dropout_1/Shape�
4trunk_/drop_3/dropout_1/random_uniform/RandomUniformRandomUniform&trunk_/drop_3/dropout_1/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype026
4trunk_/drop_3/dropout_1/random_uniform/RandomUniform�
&trunk_/drop_3/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_3/dropout_1/GreaterEqual/y�
$trunk_/drop_3/dropout_1/GreaterEqualGreaterEqual=trunk_/drop_3/dropout_1/random_uniform/RandomUniform:output:0/trunk_/drop_3/dropout_1/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2&
$trunk_/drop_3/dropout_1/GreaterEqual�
trunk_/drop_3/dropout_1/CastCast(trunk_/drop_3/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_1/Cast�
trunk_/drop_3/dropout_1/Mul_1Multrunk_/drop_3/dropout_1/Mul:z:0 trunk_/drop_3/dropout_1/Cast:y:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_1/Mul_1�
%trunk_/conv_1/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_2/ExpandDims/dim�
!trunk_/conv_1/conv1d_2/ExpandDims
ExpandDimsinputs_4.trunk_/conv_1/conv1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_2/ExpandDims�
2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_2/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_2/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_2/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_2/ExpandDims_1�
trunk_/conv_1/conv1d_2Conv2D*trunk_/conv_1/conv1d_2/ExpandDims:output:0,trunk_/conv_1/conv1d_2/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_2�
trunk_/conv_1/conv1d_2/SqueezeSqueezetrunk_/conv_1/conv1d_2:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_2/Squeeze�
&trunk_/conv_1/BiasAdd_2/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_2/ReadVariableOp�
trunk_/conv_1/BiasAdd_2BiasAdd'trunk_/conv_1/conv1d_2/Squeeze:output:0.trunk_/conv_1/BiasAdd_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_2�
trunk_/conv_1/Relu_2Relu trunk_/conv_1/BiasAdd_2:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_2�
trunk_/drop_1/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_1/dropout_2/Const�
trunk_/drop_1/dropout_2/MulMul"trunk_/conv_1/Relu_2:activations:0&trunk_/drop_1/dropout_2/Const:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_2/Mul�
trunk_/drop_1/dropout_2/ShapeShape"trunk_/conv_1/Relu_2:activations:0*
T0*
_output_shapes
:2
trunk_/drop_1/dropout_2/Shape�
4trunk_/drop_1/dropout_2/random_uniform/RandomUniformRandomUniform&trunk_/drop_1/dropout_2/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype026
4trunk_/drop_1/dropout_2/random_uniform/RandomUniform�
&trunk_/drop_1/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_1/dropout_2/GreaterEqual/y�
$trunk_/drop_1/dropout_2/GreaterEqualGreaterEqual=trunk_/drop_1/dropout_2/random_uniform/RandomUniform:output:0/trunk_/drop_1/dropout_2/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2&
$trunk_/drop_1/dropout_2/GreaterEqual�
trunk_/drop_1/dropout_2/CastCast(trunk_/drop_1/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_2/Cast�
trunk_/drop_1/dropout_2/Mul_1Multrunk_/drop_1/dropout_2/Mul:z:0 trunk_/drop_1/dropout_2/Cast:y:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_2/Mul_1�
%trunk_/conv_2/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_2/ExpandDims/dim�
!trunk_/conv_2/conv1d_2/ExpandDims
ExpandDims!trunk_/drop_1/dropout_2/Mul_1:z:0.trunk_/conv_2/conv1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_2/ExpandDims�
2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_2/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_2/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_2/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_2/ExpandDims_1�
trunk_/conv_2/conv1d_2Conv2D*trunk_/conv_2/conv1d_2/ExpandDims:output:0,trunk_/conv_2/conv1d_2/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_2�
trunk_/conv_2/conv1d_2/SqueezeSqueezetrunk_/conv_2/conv1d_2:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_2/Squeeze�
&trunk_/conv_2/BiasAdd_2/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_2/ReadVariableOp�
trunk_/conv_2/BiasAdd_2BiasAdd'trunk_/conv_2/conv1d_2/Squeeze:output:0.trunk_/conv_2/BiasAdd_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_2�
trunk_/conv_2/Relu_2Relu trunk_/conv_2/BiasAdd_2:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_2�
trunk_/drop_2/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_2/dropout_2/Const�
trunk_/drop_2/dropout_2/MulMul"trunk_/conv_2/Relu_2:activations:0&trunk_/drop_2/dropout_2/Const:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_2/Mul�
trunk_/drop_2/dropout_2/ShapeShape"trunk_/conv_2/Relu_2:activations:0*
T0*
_output_shapes
:2
trunk_/drop_2/dropout_2/Shape�
4trunk_/drop_2/dropout_2/random_uniform/RandomUniformRandomUniform&trunk_/drop_2/dropout_2/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype026
4trunk_/drop_2/dropout_2/random_uniform/RandomUniform�
&trunk_/drop_2/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_2/dropout_2/GreaterEqual/y�
$trunk_/drop_2/dropout_2/GreaterEqualGreaterEqual=trunk_/drop_2/dropout_2/random_uniform/RandomUniform:output:0/trunk_/drop_2/dropout_2/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2&
$trunk_/drop_2/dropout_2/GreaterEqual�
trunk_/drop_2/dropout_2/CastCast(trunk_/drop_2/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_2/Cast�
trunk_/drop_2/dropout_2/Mul_1Multrunk_/drop_2/dropout_2/Mul:z:0 trunk_/drop_2/dropout_2/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_2/Mul_1�
%trunk_/conv_3/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_2/ExpandDims/dim�
!trunk_/conv_3/conv1d_2/ExpandDims
ExpandDims!trunk_/drop_2/dropout_2/Mul_1:z:0.trunk_/conv_3/conv1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_2/ExpandDims�
2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_2/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_2/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_2/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_2/ExpandDims_1�
trunk_/conv_3/conv1d_2Conv2D*trunk_/conv_3/conv1d_2/ExpandDims:output:0,trunk_/conv_3/conv1d_2/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_2�
trunk_/conv_3/conv1d_2/SqueezeSqueezetrunk_/conv_3/conv1d_2:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_2/Squeeze�
&trunk_/conv_3/BiasAdd_2/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_2/ReadVariableOp�
trunk_/conv_3/BiasAdd_2BiasAdd'trunk_/conv_3/conv1d_2/Squeeze:output:0.trunk_/conv_3/BiasAdd_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_2�
trunk_/conv_3/Relu_2Relu trunk_/conv_3/BiasAdd_2:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_2�
trunk_/drop_3/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_3/dropout_2/Const�
trunk_/drop_3/dropout_2/MulMul"trunk_/conv_3/Relu_2:activations:0&trunk_/drop_3/dropout_2/Const:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_2/Mul�
trunk_/drop_3/dropout_2/ShapeShape"trunk_/conv_3/Relu_2:activations:0*
T0*
_output_shapes
:2
trunk_/drop_3/dropout_2/Shape�
4trunk_/drop_3/dropout_2/random_uniform/RandomUniformRandomUniform&trunk_/drop_3/dropout_2/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype026
4trunk_/drop_3/dropout_2/random_uniform/RandomUniform�
&trunk_/drop_3/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_3/dropout_2/GreaterEqual/y�
$trunk_/drop_3/dropout_2/GreaterEqualGreaterEqual=trunk_/drop_3/dropout_2/random_uniform/RandomUniform:output:0/trunk_/drop_3/dropout_2/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2&
$trunk_/drop_3/dropout_2/GreaterEqual�
trunk_/drop_3/dropout_2/CastCast(trunk_/drop_3/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_2/Cast�
trunk_/drop_3/dropout_2/Mul_1Multrunk_/drop_3/dropout_2/Mul:z:0 trunk_/drop_3/dropout_2/Cast:y:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_2/Mul_1�
%trunk_/conv_1/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_3/ExpandDims/dim�
!trunk_/conv_1/conv1d_3/ExpandDims
ExpandDimsinputs_3.trunk_/conv_1/conv1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_3/ExpandDims�
2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_3/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_3/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_3/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_3/ExpandDims_1�
trunk_/conv_1/conv1d_3Conv2D*trunk_/conv_1/conv1d_3/ExpandDims:output:0,trunk_/conv_1/conv1d_3/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_3�
trunk_/conv_1/conv1d_3/SqueezeSqueezetrunk_/conv_1/conv1d_3:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_3/Squeeze�
&trunk_/conv_1/BiasAdd_3/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_3/ReadVariableOp�
trunk_/conv_1/BiasAdd_3BiasAdd'trunk_/conv_1/conv1d_3/Squeeze:output:0.trunk_/conv_1/BiasAdd_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_3�
trunk_/conv_1/Relu_3Relu trunk_/conv_1/BiasAdd_3:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_3�
trunk_/drop_1/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_1/dropout_3/Const�
trunk_/drop_1/dropout_3/MulMul"trunk_/conv_1/Relu_3:activations:0&trunk_/drop_1/dropout_3/Const:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_3/Mul�
trunk_/drop_1/dropout_3/ShapeShape"trunk_/conv_1/Relu_3:activations:0*
T0*
_output_shapes
:2
trunk_/drop_1/dropout_3/Shape�
4trunk_/drop_1/dropout_3/random_uniform/RandomUniformRandomUniform&trunk_/drop_1/dropout_3/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype026
4trunk_/drop_1/dropout_3/random_uniform/RandomUniform�
&trunk_/drop_1/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_1/dropout_3/GreaterEqual/y�
$trunk_/drop_1/dropout_3/GreaterEqualGreaterEqual=trunk_/drop_1/dropout_3/random_uniform/RandomUniform:output:0/trunk_/drop_1/dropout_3/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2&
$trunk_/drop_1/dropout_3/GreaterEqual�
trunk_/drop_1/dropout_3/CastCast(trunk_/drop_1/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_3/Cast�
trunk_/drop_1/dropout_3/Mul_1Multrunk_/drop_1/dropout_3/Mul:z:0 trunk_/drop_1/dropout_3/Cast:y:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_3/Mul_1�
%trunk_/conv_2/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_3/ExpandDims/dim�
!trunk_/conv_2/conv1d_3/ExpandDims
ExpandDims!trunk_/drop_1/dropout_3/Mul_1:z:0.trunk_/conv_2/conv1d_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_3/ExpandDims�
2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_3/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_3/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_3/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_3/ExpandDims_1�
trunk_/conv_2/conv1d_3Conv2D*trunk_/conv_2/conv1d_3/ExpandDims:output:0,trunk_/conv_2/conv1d_3/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_3�
trunk_/conv_2/conv1d_3/SqueezeSqueezetrunk_/conv_2/conv1d_3:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_3/Squeeze�
&trunk_/conv_2/BiasAdd_3/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_3/ReadVariableOp�
trunk_/conv_2/BiasAdd_3BiasAdd'trunk_/conv_2/conv1d_3/Squeeze:output:0.trunk_/conv_2/BiasAdd_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_3�
trunk_/conv_2/Relu_3Relu trunk_/conv_2/BiasAdd_3:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_3�
trunk_/drop_2/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_2/dropout_3/Const�
trunk_/drop_2/dropout_3/MulMul"trunk_/conv_2/Relu_3:activations:0&trunk_/drop_2/dropout_3/Const:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_3/Mul�
trunk_/drop_2/dropout_3/ShapeShape"trunk_/conv_2/Relu_3:activations:0*
T0*
_output_shapes
:2
trunk_/drop_2/dropout_3/Shape�
4trunk_/drop_2/dropout_3/random_uniform/RandomUniformRandomUniform&trunk_/drop_2/dropout_3/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype026
4trunk_/drop_2/dropout_3/random_uniform/RandomUniform�
&trunk_/drop_2/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_2/dropout_3/GreaterEqual/y�
$trunk_/drop_2/dropout_3/GreaterEqualGreaterEqual=trunk_/drop_2/dropout_3/random_uniform/RandomUniform:output:0/trunk_/drop_2/dropout_3/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2&
$trunk_/drop_2/dropout_3/GreaterEqual�
trunk_/drop_2/dropout_3/CastCast(trunk_/drop_2/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_3/Cast�
trunk_/drop_2/dropout_3/Mul_1Multrunk_/drop_2/dropout_3/Mul:z:0 trunk_/drop_2/dropout_3/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_3/Mul_1�
%trunk_/conv_3/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_3/ExpandDims/dim�
!trunk_/conv_3/conv1d_3/ExpandDims
ExpandDims!trunk_/drop_2/dropout_3/Mul_1:z:0.trunk_/conv_3/conv1d_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_3/ExpandDims�
2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_3/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_3/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_3/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_3/ExpandDims_1�
trunk_/conv_3/conv1d_3Conv2D*trunk_/conv_3/conv1d_3/ExpandDims:output:0,trunk_/conv_3/conv1d_3/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_3�
trunk_/conv_3/conv1d_3/SqueezeSqueezetrunk_/conv_3/conv1d_3:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_3/Squeeze�
&trunk_/conv_3/BiasAdd_3/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_3/ReadVariableOp�
trunk_/conv_3/BiasAdd_3BiasAdd'trunk_/conv_3/conv1d_3/Squeeze:output:0.trunk_/conv_3/BiasAdd_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_3�
trunk_/conv_3/Relu_3Relu trunk_/conv_3/BiasAdd_3:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_3�
trunk_/drop_3/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_3/dropout_3/Const�
trunk_/drop_3/dropout_3/MulMul"trunk_/conv_3/Relu_3:activations:0&trunk_/drop_3/dropout_3/Const:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_3/Mul�
trunk_/drop_3/dropout_3/ShapeShape"trunk_/conv_3/Relu_3:activations:0*
T0*
_output_shapes
:2
trunk_/drop_3/dropout_3/Shape�
4trunk_/drop_3/dropout_3/random_uniform/RandomUniformRandomUniform&trunk_/drop_3/dropout_3/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype026
4trunk_/drop_3/dropout_3/random_uniform/RandomUniform�
&trunk_/drop_3/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_3/dropout_3/GreaterEqual/y�
$trunk_/drop_3/dropout_3/GreaterEqualGreaterEqual=trunk_/drop_3/dropout_3/random_uniform/RandomUniform:output:0/trunk_/drop_3/dropout_3/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2&
$trunk_/drop_3/dropout_3/GreaterEqual�
trunk_/drop_3/dropout_3/CastCast(trunk_/drop_3/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_3/Cast�
trunk_/drop_3/dropout_3/Mul_1Multrunk_/drop_3/dropout_3/Mul:z:0 trunk_/drop_3/dropout_3/Cast:y:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_3/Mul_1�
%trunk_/conv_1/conv1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_4/ExpandDims/dim�
!trunk_/conv_1/conv1d_4/ExpandDims
ExpandDimsinputs_2.trunk_/conv_1/conv1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_4/ExpandDims�
2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_4/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_4/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_4/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_4/ExpandDims_1�
trunk_/conv_1/conv1d_4Conv2D*trunk_/conv_1/conv1d_4/ExpandDims:output:0,trunk_/conv_1/conv1d_4/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_4�
trunk_/conv_1/conv1d_4/SqueezeSqueezetrunk_/conv_1/conv1d_4:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_4/Squeeze�
&trunk_/conv_1/BiasAdd_4/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_4/ReadVariableOp�
trunk_/conv_1/BiasAdd_4BiasAdd'trunk_/conv_1/conv1d_4/Squeeze:output:0.trunk_/conv_1/BiasAdd_4/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_4�
trunk_/conv_1/Relu_4Relu trunk_/conv_1/BiasAdd_4:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_4�
trunk_/drop_1/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_1/dropout_4/Const�
trunk_/drop_1/dropout_4/MulMul"trunk_/conv_1/Relu_4:activations:0&trunk_/drop_1/dropout_4/Const:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_4/Mul�
trunk_/drop_1/dropout_4/ShapeShape"trunk_/conv_1/Relu_4:activations:0*
T0*
_output_shapes
:2
trunk_/drop_1/dropout_4/Shape�
4trunk_/drop_1/dropout_4/random_uniform/RandomUniformRandomUniform&trunk_/drop_1/dropout_4/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype026
4trunk_/drop_1/dropout_4/random_uniform/RandomUniform�
&trunk_/drop_1/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_1/dropout_4/GreaterEqual/y�
$trunk_/drop_1/dropout_4/GreaterEqualGreaterEqual=trunk_/drop_1/dropout_4/random_uniform/RandomUniform:output:0/trunk_/drop_1/dropout_4/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2&
$trunk_/drop_1/dropout_4/GreaterEqual�
trunk_/drop_1/dropout_4/CastCast(trunk_/drop_1/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_4/Cast�
trunk_/drop_1/dropout_4/Mul_1Multrunk_/drop_1/dropout_4/Mul:z:0 trunk_/drop_1/dropout_4/Cast:y:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_4/Mul_1�
%trunk_/conv_2/conv1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_4/ExpandDims/dim�
!trunk_/conv_2/conv1d_4/ExpandDims
ExpandDims!trunk_/drop_1/dropout_4/Mul_1:z:0.trunk_/conv_2/conv1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_4/ExpandDims�
2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_4/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_4/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_4/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_4/ExpandDims_1�
trunk_/conv_2/conv1d_4Conv2D*trunk_/conv_2/conv1d_4/ExpandDims:output:0,trunk_/conv_2/conv1d_4/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_4�
trunk_/conv_2/conv1d_4/SqueezeSqueezetrunk_/conv_2/conv1d_4:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_4/Squeeze�
&trunk_/conv_2/BiasAdd_4/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_4/ReadVariableOp�
trunk_/conv_2/BiasAdd_4BiasAdd'trunk_/conv_2/conv1d_4/Squeeze:output:0.trunk_/conv_2/BiasAdd_4/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_4�
trunk_/conv_2/Relu_4Relu trunk_/conv_2/BiasAdd_4:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_4�
trunk_/drop_2/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_2/dropout_4/Const�
trunk_/drop_2/dropout_4/MulMul"trunk_/conv_2/Relu_4:activations:0&trunk_/drop_2/dropout_4/Const:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_4/Mul�
trunk_/drop_2/dropout_4/ShapeShape"trunk_/conv_2/Relu_4:activations:0*
T0*
_output_shapes
:2
trunk_/drop_2/dropout_4/Shape�
4trunk_/drop_2/dropout_4/random_uniform/RandomUniformRandomUniform&trunk_/drop_2/dropout_4/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype026
4trunk_/drop_2/dropout_4/random_uniform/RandomUniform�
&trunk_/drop_2/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_2/dropout_4/GreaterEqual/y�
$trunk_/drop_2/dropout_4/GreaterEqualGreaterEqual=trunk_/drop_2/dropout_4/random_uniform/RandomUniform:output:0/trunk_/drop_2/dropout_4/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2&
$trunk_/drop_2/dropout_4/GreaterEqual�
trunk_/drop_2/dropout_4/CastCast(trunk_/drop_2/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_4/Cast�
trunk_/drop_2/dropout_4/Mul_1Multrunk_/drop_2/dropout_4/Mul:z:0 trunk_/drop_2/dropout_4/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_4/Mul_1�
%trunk_/conv_3/conv1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_4/ExpandDims/dim�
!trunk_/conv_3/conv1d_4/ExpandDims
ExpandDims!trunk_/drop_2/dropout_4/Mul_1:z:0.trunk_/conv_3/conv1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_4/ExpandDims�
2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_4/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_4/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_4/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_4/ExpandDims_1�
trunk_/conv_3/conv1d_4Conv2D*trunk_/conv_3/conv1d_4/ExpandDims:output:0,trunk_/conv_3/conv1d_4/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_4�
trunk_/conv_3/conv1d_4/SqueezeSqueezetrunk_/conv_3/conv1d_4:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_4/Squeeze�
&trunk_/conv_3/BiasAdd_4/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_4/ReadVariableOp�
trunk_/conv_3/BiasAdd_4BiasAdd'trunk_/conv_3/conv1d_4/Squeeze:output:0.trunk_/conv_3/BiasAdd_4/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_4�
trunk_/conv_3/Relu_4Relu trunk_/conv_3/BiasAdd_4:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_4�
trunk_/drop_3/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_3/dropout_4/Const�
trunk_/drop_3/dropout_4/MulMul"trunk_/conv_3/Relu_4:activations:0&trunk_/drop_3/dropout_4/Const:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_4/Mul�
trunk_/drop_3/dropout_4/ShapeShape"trunk_/conv_3/Relu_4:activations:0*
T0*
_output_shapes
:2
trunk_/drop_3/dropout_4/Shape�
4trunk_/drop_3/dropout_4/random_uniform/RandomUniformRandomUniform&trunk_/drop_3/dropout_4/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype026
4trunk_/drop_3/dropout_4/random_uniform/RandomUniform�
&trunk_/drop_3/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_3/dropout_4/GreaterEqual/y�
$trunk_/drop_3/dropout_4/GreaterEqualGreaterEqual=trunk_/drop_3/dropout_4/random_uniform/RandomUniform:output:0/trunk_/drop_3/dropout_4/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2&
$trunk_/drop_3/dropout_4/GreaterEqual�
trunk_/drop_3/dropout_4/CastCast(trunk_/drop_3/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_4/Cast�
trunk_/drop_3/dropout_4/Mul_1Multrunk_/drop_3/dropout_4/Mul:z:0 trunk_/drop_3/dropout_4/Cast:y:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_4/Mul_1�
%trunk_/conv_1/conv1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_5/ExpandDims/dim�
!trunk_/conv_1/conv1d_5/ExpandDims
ExpandDimsinputs_1.trunk_/conv_1/conv1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_5/ExpandDims�
2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_5/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_5/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_5/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_5/ExpandDims_1�
trunk_/conv_1/conv1d_5Conv2D*trunk_/conv_1/conv1d_5/ExpandDims:output:0,trunk_/conv_1/conv1d_5/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_5�
trunk_/conv_1/conv1d_5/SqueezeSqueezetrunk_/conv_1/conv1d_5:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_5/Squeeze�
&trunk_/conv_1/BiasAdd_5/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_5/ReadVariableOp�
trunk_/conv_1/BiasAdd_5BiasAdd'trunk_/conv_1/conv1d_5/Squeeze:output:0.trunk_/conv_1/BiasAdd_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_5�
trunk_/conv_1/Relu_5Relu trunk_/conv_1/BiasAdd_5:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_5�
trunk_/drop_1/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_1/dropout_5/Const�
trunk_/drop_1/dropout_5/MulMul"trunk_/conv_1/Relu_5:activations:0&trunk_/drop_1/dropout_5/Const:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_5/Mul�
trunk_/drop_1/dropout_5/ShapeShape"trunk_/conv_1/Relu_5:activations:0*
T0*
_output_shapes
:2
trunk_/drop_1/dropout_5/Shape�
4trunk_/drop_1/dropout_5/random_uniform/RandomUniformRandomUniform&trunk_/drop_1/dropout_5/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype026
4trunk_/drop_1/dropout_5/random_uniform/RandomUniform�
&trunk_/drop_1/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_1/dropout_5/GreaterEqual/y�
$trunk_/drop_1/dropout_5/GreaterEqualGreaterEqual=trunk_/drop_1/dropout_5/random_uniform/RandomUniform:output:0/trunk_/drop_1/dropout_5/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2&
$trunk_/drop_1/dropout_5/GreaterEqual�
trunk_/drop_1/dropout_5/CastCast(trunk_/drop_1/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_5/Cast�
trunk_/drop_1/dropout_5/Mul_1Multrunk_/drop_1/dropout_5/Mul:z:0 trunk_/drop_1/dropout_5/Cast:y:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_5/Mul_1�
%trunk_/conv_2/conv1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_5/ExpandDims/dim�
!trunk_/conv_2/conv1d_5/ExpandDims
ExpandDims!trunk_/drop_1/dropout_5/Mul_1:z:0.trunk_/conv_2/conv1d_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_5/ExpandDims�
2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_5/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_5/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_5/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_5/ExpandDims_1�
trunk_/conv_2/conv1d_5Conv2D*trunk_/conv_2/conv1d_5/ExpandDims:output:0,trunk_/conv_2/conv1d_5/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_5�
trunk_/conv_2/conv1d_5/SqueezeSqueezetrunk_/conv_2/conv1d_5:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_5/Squeeze�
&trunk_/conv_2/BiasAdd_5/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_5/ReadVariableOp�
trunk_/conv_2/BiasAdd_5BiasAdd'trunk_/conv_2/conv1d_5/Squeeze:output:0.trunk_/conv_2/BiasAdd_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_5�
trunk_/conv_2/Relu_5Relu trunk_/conv_2/BiasAdd_5:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_5�
trunk_/drop_2/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_2/dropout_5/Const�
trunk_/drop_2/dropout_5/MulMul"trunk_/conv_2/Relu_5:activations:0&trunk_/drop_2/dropout_5/Const:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_5/Mul�
trunk_/drop_2/dropout_5/ShapeShape"trunk_/conv_2/Relu_5:activations:0*
T0*
_output_shapes
:2
trunk_/drop_2/dropout_5/Shape�
4trunk_/drop_2/dropout_5/random_uniform/RandomUniformRandomUniform&trunk_/drop_2/dropout_5/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype026
4trunk_/drop_2/dropout_5/random_uniform/RandomUniform�
&trunk_/drop_2/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_2/dropout_5/GreaterEqual/y�
$trunk_/drop_2/dropout_5/GreaterEqualGreaterEqual=trunk_/drop_2/dropout_5/random_uniform/RandomUniform:output:0/trunk_/drop_2/dropout_5/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2&
$trunk_/drop_2/dropout_5/GreaterEqual�
trunk_/drop_2/dropout_5/CastCast(trunk_/drop_2/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_5/Cast�
trunk_/drop_2/dropout_5/Mul_1Multrunk_/drop_2/dropout_5/Mul:z:0 trunk_/drop_2/dropout_5/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_5/Mul_1�
%trunk_/conv_3/conv1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_5/ExpandDims/dim�
!trunk_/conv_3/conv1d_5/ExpandDims
ExpandDims!trunk_/drop_2/dropout_5/Mul_1:z:0.trunk_/conv_3/conv1d_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_5/ExpandDims�
2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_5/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_5/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_5/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_5/ExpandDims_1�
trunk_/conv_3/conv1d_5Conv2D*trunk_/conv_3/conv1d_5/ExpandDims:output:0,trunk_/conv_3/conv1d_5/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_5�
trunk_/conv_3/conv1d_5/SqueezeSqueezetrunk_/conv_3/conv1d_5:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_5/Squeeze�
&trunk_/conv_3/BiasAdd_5/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_5/ReadVariableOp�
trunk_/conv_3/BiasAdd_5BiasAdd'trunk_/conv_3/conv1d_5/Squeeze:output:0.trunk_/conv_3/BiasAdd_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_5�
trunk_/conv_3/Relu_5Relu trunk_/conv_3/BiasAdd_5:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_5�
trunk_/drop_3/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_3/dropout_5/Const�
trunk_/drop_3/dropout_5/MulMul"trunk_/conv_3/Relu_5:activations:0&trunk_/drop_3/dropout_5/Const:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_5/Mul�
trunk_/drop_3/dropout_5/ShapeShape"trunk_/conv_3/Relu_5:activations:0*
T0*
_output_shapes
:2
trunk_/drop_3/dropout_5/Shape�
4trunk_/drop_3/dropout_5/random_uniform/RandomUniformRandomUniform&trunk_/drop_3/dropout_5/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype026
4trunk_/drop_3/dropout_5/random_uniform/RandomUniform�
&trunk_/drop_3/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_3/dropout_5/GreaterEqual/y�
$trunk_/drop_3/dropout_5/GreaterEqualGreaterEqual=trunk_/drop_3/dropout_5/random_uniform/RandomUniform:output:0/trunk_/drop_3/dropout_5/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2&
$trunk_/drop_3/dropout_5/GreaterEqual�
trunk_/drop_3/dropout_5/CastCast(trunk_/drop_3/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_5/Cast�
trunk_/drop_3/dropout_5/Mul_1Multrunk_/drop_3/dropout_5/Mul:z:0 trunk_/drop_3/dropout_5/Cast:y:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_5/Mul_1�
%trunk_/conv_1/conv1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_6/ExpandDims/dim�
!trunk_/conv_1/conv1d_6/ExpandDims
ExpandDimsinputs_0.trunk_/conv_1/conv1d_6/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_6/ExpandDims�
2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_6/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_6/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_6/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_6/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_6/ExpandDims_1�
trunk_/conv_1/conv1d_6Conv2D*trunk_/conv_1/conv1d_6/ExpandDims:output:0,trunk_/conv_1/conv1d_6/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_6�
trunk_/conv_1/conv1d_6/SqueezeSqueezetrunk_/conv_1/conv1d_6:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_6/Squeeze�
&trunk_/conv_1/BiasAdd_6/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_6/ReadVariableOp�
trunk_/conv_1/BiasAdd_6BiasAdd'trunk_/conv_1/conv1d_6/Squeeze:output:0.trunk_/conv_1/BiasAdd_6/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_6�
trunk_/conv_1/Relu_6Relu trunk_/conv_1/BiasAdd_6:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_6�
trunk_/drop_1/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_1/dropout_6/Const�
trunk_/drop_1/dropout_6/MulMul"trunk_/conv_1/Relu_6:activations:0&trunk_/drop_1/dropout_6/Const:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_6/Mul�
trunk_/drop_1/dropout_6/ShapeShape"trunk_/conv_1/Relu_6:activations:0*
T0*
_output_shapes
:2
trunk_/drop_1/dropout_6/Shape�
4trunk_/drop_1/dropout_6/random_uniform/RandomUniformRandomUniform&trunk_/drop_1/dropout_6/Shape:output:0*
T0*+
_output_shapes
:���������i *
dtype026
4trunk_/drop_1/dropout_6/random_uniform/RandomUniform�
&trunk_/drop_1/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_1/dropout_6/GreaterEqual/y�
$trunk_/drop_1/dropout_6/GreaterEqualGreaterEqual=trunk_/drop_1/dropout_6/random_uniform/RandomUniform:output:0/trunk_/drop_1/dropout_6/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������i 2&
$trunk_/drop_1/dropout_6/GreaterEqual�
trunk_/drop_1/dropout_6/CastCast(trunk_/drop_1/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_6/Cast�
trunk_/drop_1/dropout_6/Mul_1Multrunk_/drop_1/dropout_6/Mul:z:0 trunk_/drop_1/dropout_6/Cast:y:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/dropout_6/Mul_1�
%trunk_/conv_2/conv1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_6/ExpandDims/dim�
!trunk_/conv_2/conv1d_6/ExpandDims
ExpandDims!trunk_/drop_1/dropout_6/Mul_1:z:0.trunk_/conv_2/conv1d_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_6/ExpandDims�
2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_6/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_6/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_6/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_6/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_6/ExpandDims_1�
trunk_/conv_2/conv1d_6Conv2D*trunk_/conv_2/conv1d_6/ExpandDims:output:0,trunk_/conv_2/conv1d_6/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_6�
trunk_/conv_2/conv1d_6/SqueezeSqueezetrunk_/conv_2/conv1d_6:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_6/Squeeze�
&trunk_/conv_2/BiasAdd_6/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_6/ReadVariableOp�
trunk_/conv_2/BiasAdd_6BiasAdd'trunk_/conv_2/conv1d_6/Squeeze:output:0.trunk_/conv_2/BiasAdd_6/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_6�
trunk_/conv_2/Relu_6Relu trunk_/conv_2/BiasAdd_6:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_6�
trunk_/drop_2/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_2/dropout_6/Const�
trunk_/drop_2/dropout_6/MulMul"trunk_/conv_2/Relu_6:activations:0&trunk_/drop_2/dropout_6/Const:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_6/Mul�
trunk_/drop_2/dropout_6/ShapeShape"trunk_/conv_2/Relu_6:activations:0*
T0*
_output_shapes
:2
trunk_/drop_2/dropout_6/Shape�
4trunk_/drop_2/dropout_6/random_uniform/RandomUniformRandomUniform&trunk_/drop_2/dropout_6/Shape:output:0*
T0*+
_output_shapes
:���������Z@*
dtype026
4trunk_/drop_2/dropout_6/random_uniform/RandomUniform�
&trunk_/drop_2/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_2/dropout_6/GreaterEqual/y�
$trunk_/drop_2/dropout_6/GreaterEqualGreaterEqual=trunk_/drop_2/dropout_6/random_uniform/RandomUniform:output:0/trunk_/drop_2/dropout_6/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������Z@2&
$trunk_/drop_2/dropout_6/GreaterEqual�
trunk_/drop_2/dropout_6/CastCast(trunk_/drop_2/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_6/Cast�
trunk_/drop_2/dropout_6/Mul_1Multrunk_/drop_2/dropout_6/Mul:z:0 trunk_/drop_2/dropout_6/Cast:y:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/dropout_6/Mul_1�
%trunk_/conv_3/conv1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_6/ExpandDims/dim�
!trunk_/conv_3/conv1d_6/ExpandDims
ExpandDims!trunk_/drop_2/dropout_6/Mul_1:z:0.trunk_/conv_3/conv1d_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_6/ExpandDims�
2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_6/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_6/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_6/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_6/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_6/ExpandDims_1�
trunk_/conv_3/conv1d_6Conv2D*trunk_/conv_3/conv1d_6/ExpandDims:output:0,trunk_/conv_3/conv1d_6/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_6�
trunk_/conv_3/conv1d_6/SqueezeSqueezetrunk_/conv_3/conv1d_6:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_6/Squeeze�
&trunk_/conv_3/BiasAdd_6/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_6/ReadVariableOp�
trunk_/conv_3/BiasAdd_6BiasAdd'trunk_/conv_3/conv1d_6/Squeeze:output:0.trunk_/conv_3/BiasAdd_6/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_6�
trunk_/conv_3/Relu_6Relu trunk_/conv_3/BiasAdd_6:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_6�
trunk_/drop_3/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
trunk_/drop_3/dropout_6/Const�
trunk_/drop_3/dropout_6/MulMul"trunk_/conv_3/Relu_6:activations:0&trunk_/drop_3/dropout_6/Const:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_6/Mul�
trunk_/drop_3/dropout_6/ShapeShape"trunk_/conv_3/Relu_6:activations:0*
T0*
_output_shapes
:2
trunk_/drop_3/dropout_6/Shape�
4trunk_/drop_3/dropout_6/random_uniform/RandomUniformRandomUniform&trunk_/drop_3/dropout_6/Shape:output:0*
T0*+
_output_shapes
:���������S`*
dtype026
4trunk_/drop_3/dropout_6/random_uniform/RandomUniform�
&trunk_/drop_3/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2(
&trunk_/drop_3/dropout_6/GreaterEqual/y�
$trunk_/drop_3/dropout_6/GreaterEqualGreaterEqual=trunk_/drop_3/dropout_6/random_uniform/RandomUniform:output:0/trunk_/drop_3/dropout_6/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������S`2&
$trunk_/drop_3/dropout_6/GreaterEqual�
trunk_/drop_3/dropout_6/CastCast(trunk_/drop_3/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_6/Cast�
trunk_/drop_3/dropout_6/Mul_1Multrunk_/drop_3/dropout_6/Mul:z:0 trunk_/drop_3/dropout_6/Cast:y:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/dropout_6/Mul_1�
%global_max_pool_/pool_/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%global_max_pool_/pool_/ExpandDims/dim�
!global_max_pool_/pool_/ExpandDims
ExpandDimstrunk_/drop_3/dropout/Mul_1:z:0.global_max_pool_/pool_/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������S`2#
!global_max_pool_/pool_/ExpandDims�
global_max_pool_/pool_/MaxPoolMaxPool*global_max_pool_/pool_/ExpandDims:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2 
global_max_pool_/pool_/MaxPool�
global_max_pool_/pool_/SqueezeSqueeze'global_max_pool_/pool_/MaxPool:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2 
global_max_pool_/pool_/Squeeze�
global_max_pool_/flat_/ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
global_max_pool_/flat_/Const�
global_max_pool_/flat_/ReshapeReshape'global_max_pool_/pool_/Squeeze:output:0%global_max_pool_/flat_/Const:output:0*
T0*(
_output_shapes
:����������2 
global_max_pool_/flat_/Reshape�
'global_max_pool_/pool_/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_1/dim�
#global_max_pool_/pool_/ExpandDims_1
ExpandDims!trunk_/drop_3/dropout_1/Mul_1:z:00global_max_pool_/pool_/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_1�
 global_max_pool_/pool_/MaxPool_1MaxPool,global_max_pool_/pool_/ExpandDims_1:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_1�
 global_max_pool_/pool_/Squeeze_1Squeeze)global_max_pool_/pool_/MaxPool_1:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_1�
global_max_pool_/flat_/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_1�
 global_max_pool_/flat_/Reshape_1Reshape)global_max_pool_/pool_/Squeeze_1:output:0'global_max_pool_/flat_/Const_1:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_1�
'global_max_pool_/pool_/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_2/dim�
#global_max_pool_/pool_/ExpandDims_2
ExpandDims!trunk_/drop_3/dropout_2/Mul_1:z:00global_max_pool_/pool_/ExpandDims_2/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_2�
 global_max_pool_/pool_/MaxPool_2MaxPool,global_max_pool_/pool_/ExpandDims_2:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_2�
 global_max_pool_/pool_/Squeeze_2Squeeze)global_max_pool_/pool_/MaxPool_2:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_2�
global_max_pool_/flat_/Const_2Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_2�
 global_max_pool_/flat_/Reshape_2Reshape)global_max_pool_/pool_/Squeeze_2:output:0'global_max_pool_/flat_/Const_2:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_2�
'global_max_pool_/pool_/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_3/dim�
#global_max_pool_/pool_/ExpandDims_3
ExpandDims!trunk_/drop_3/dropout_3/Mul_1:z:00global_max_pool_/pool_/ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_3�
 global_max_pool_/pool_/MaxPool_3MaxPool,global_max_pool_/pool_/ExpandDims_3:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_3�
 global_max_pool_/pool_/Squeeze_3Squeeze)global_max_pool_/pool_/MaxPool_3:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_3�
global_max_pool_/flat_/Const_3Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_3�
 global_max_pool_/flat_/Reshape_3Reshape)global_max_pool_/pool_/Squeeze_3:output:0'global_max_pool_/flat_/Const_3:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_3�
'global_max_pool_/pool_/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_4/dim�
#global_max_pool_/pool_/ExpandDims_4
ExpandDims!trunk_/drop_3/dropout_4/Mul_1:z:00global_max_pool_/pool_/ExpandDims_4/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_4�
 global_max_pool_/pool_/MaxPool_4MaxPool,global_max_pool_/pool_/ExpandDims_4:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_4�
 global_max_pool_/pool_/Squeeze_4Squeeze)global_max_pool_/pool_/MaxPool_4:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_4�
global_max_pool_/flat_/Const_4Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_4�
 global_max_pool_/flat_/Reshape_4Reshape)global_max_pool_/pool_/Squeeze_4:output:0'global_max_pool_/flat_/Const_4:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_4�
'global_max_pool_/pool_/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_5/dim�
#global_max_pool_/pool_/ExpandDims_5
ExpandDims!trunk_/drop_3/dropout_5/Mul_1:z:00global_max_pool_/pool_/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_5�
 global_max_pool_/pool_/MaxPool_5MaxPool,global_max_pool_/pool_/ExpandDims_5:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_5�
 global_max_pool_/pool_/Squeeze_5Squeeze)global_max_pool_/pool_/MaxPool_5:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_5�
global_max_pool_/flat_/Const_5Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_5�
 global_max_pool_/flat_/Reshape_5Reshape)global_max_pool_/pool_/Squeeze_5:output:0'global_max_pool_/flat_/Const_5:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_5�
'global_max_pool_/pool_/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_6/dim�
#global_max_pool_/pool_/ExpandDims_6
ExpandDims!trunk_/drop_3/dropout_6/Mul_1:z:00global_max_pool_/pool_/ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_6�
 global_max_pool_/pool_/MaxPool_6MaxPool,global_max_pool_/pool_/ExpandDims_6:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_6�
 global_max_pool_/pool_/Squeeze_6Squeeze)global_max_pool_/pool_/MaxPool_6:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_6�
global_max_pool_/flat_/Const_6Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_6�
 global_max_pool_/flat_/Reshape_6Reshape)global_max_pool_/pool_/Squeeze_6:output:0'global_max_pool_/flat_/Const_6:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_6�
dens_7/MatMul/ReadVariableOpReadVariableOp%dens_7_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_7/MatMul/ReadVariableOp�
dens_7/MatMulMatMul'global_max_pool_/flat_/Reshape:output:0$dens_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_7/MatMul�
dens_7/BiasAdd/ReadVariableOpReadVariableOp&dens_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_7/BiasAdd/ReadVariableOp�
dens_7/BiasAddBiasAdddens_7/MatMul:product:0%dens_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_7/BiasAddn
dens_7/ReluReludens_7/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_7/Relu�
dens_6/MatMul/ReadVariableOpReadVariableOp%dens_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_6/MatMul/ReadVariableOp�
dens_6/MatMulMatMul)global_max_pool_/flat_/Reshape_1:output:0$dens_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_6/MatMul�
dens_6/BiasAdd/ReadVariableOpReadVariableOp&dens_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_6/BiasAdd/ReadVariableOp�
dens_6/BiasAddBiasAdddens_6/MatMul:product:0%dens_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_6/BiasAddn
dens_6/ReluReludens_6/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_6/Relu�
dens_5/MatMul/ReadVariableOpReadVariableOp%dens_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_5/MatMul/ReadVariableOp�
dens_5/MatMulMatMul)global_max_pool_/flat_/Reshape_2:output:0$dens_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_5/MatMul�
dens_5/BiasAdd/ReadVariableOpReadVariableOp&dens_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_5/BiasAdd/ReadVariableOp�
dens_5/BiasAddBiasAdddens_5/MatMul:product:0%dens_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_5/BiasAddn
dens_5/ReluReludens_5/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_5/Relu�
dens_4/MatMul/ReadVariableOpReadVariableOp%dens_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_4/MatMul/ReadVariableOp�
dens_4/MatMulMatMul)global_max_pool_/flat_/Reshape_3:output:0$dens_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_4/MatMul�
dens_4/BiasAdd/ReadVariableOpReadVariableOp&dens_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_4/BiasAdd/ReadVariableOp�
dens_4/BiasAddBiasAdddens_4/MatMul:product:0%dens_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_4/BiasAddn
dens_4/ReluReludens_4/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_4/Relu�
dens_3/MatMul/ReadVariableOpReadVariableOp%dens_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_3/MatMul/ReadVariableOp�
dens_3/MatMulMatMul)global_max_pool_/flat_/Reshape_4:output:0$dens_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_3/MatMul�
dens_3/BiasAdd/ReadVariableOpReadVariableOp&dens_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_3/BiasAdd/ReadVariableOp�
dens_3/BiasAddBiasAdddens_3/MatMul:product:0%dens_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_3/BiasAddn
dens_3/ReluReludens_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_3/Relu�
dens_2/MatMul/ReadVariableOpReadVariableOp%dens_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_2/MatMul/ReadVariableOp�
dens_2/MatMulMatMul)global_max_pool_/flat_/Reshape_5:output:0$dens_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_2/MatMul�
dens_2/BiasAdd/ReadVariableOpReadVariableOp&dens_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_2/BiasAdd/ReadVariableOp�
dens_2/BiasAddBiasAdddens_2/MatMul:product:0%dens_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_2/BiasAddn
dens_2/ReluReludens_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_2/Relu�
dens_1/MatMul/ReadVariableOpReadVariableOp%dens_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_1/MatMul/ReadVariableOp�
dens_1/MatMulMatMul)global_max_pool_/flat_/Reshape_6:output:0$dens_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_1/MatMul�
dens_1/BiasAdd/ReadVariableOpReadVariableOp&dens_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_1/BiasAdd/ReadVariableOp�
dens_1/BiasAddBiasAdddens_1/MatMul:product:0%dens_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_1/BiasAddn
dens_1/ReluReludens_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_1/Relu�
head_7/MatMul/ReadVariableOpReadVariableOp%head_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_7/MatMul/ReadVariableOp�
head_7/MatMulMatMuldens_7/Relu:activations:0$head_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_7/MatMul�
head_7/BiasAdd/ReadVariableOpReadVariableOp&head_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_7/BiasAdd/ReadVariableOp�
head_7/BiasAddBiasAddhead_7/MatMul:product:0%head_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_7/BiasAddv
head_7/SigmoidSigmoidhead_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_7/Sigmoid�
head_6/MatMul/ReadVariableOpReadVariableOp%head_6_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_6/MatMul/ReadVariableOp�
head_6/MatMulMatMuldens_6/Relu:activations:0$head_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_6/MatMul�
head_6/BiasAdd/ReadVariableOpReadVariableOp&head_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_6/BiasAdd/ReadVariableOp�
head_6/BiasAddBiasAddhead_6/MatMul:product:0%head_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_6/BiasAddv
head_6/SigmoidSigmoidhead_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_6/Sigmoid�
head_5/MatMul/ReadVariableOpReadVariableOp%head_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_5/MatMul/ReadVariableOp�
head_5/MatMulMatMuldens_5/Relu:activations:0$head_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_5/MatMul�
head_5/BiasAdd/ReadVariableOpReadVariableOp&head_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_5/BiasAdd/ReadVariableOp�
head_5/BiasAddBiasAddhead_5/MatMul:product:0%head_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_5/BiasAddv
head_5/SigmoidSigmoidhead_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_5/Sigmoid�
head_4/MatMul/ReadVariableOpReadVariableOp%head_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_4/MatMul/ReadVariableOp�
head_4/MatMulMatMuldens_4/Relu:activations:0$head_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_4/MatMul�
head_4/BiasAdd/ReadVariableOpReadVariableOp&head_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_4/BiasAdd/ReadVariableOp�
head_4/BiasAddBiasAddhead_4/MatMul:product:0%head_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_4/BiasAddv
head_4/SigmoidSigmoidhead_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_4/Sigmoid�
head_3/MatMul/ReadVariableOpReadVariableOp%head_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_3/MatMul/ReadVariableOp�
head_3/MatMulMatMuldens_3/Relu:activations:0$head_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_3/MatMul�
head_3/BiasAdd/ReadVariableOpReadVariableOp&head_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_3/BiasAdd/ReadVariableOp�
head_3/BiasAddBiasAddhead_3/MatMul:product:0%head_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_3/BiasAddv
head_3/SigmoidSigmoidhead_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_3/Sigmoid�
head_2/MatMul/ReadVariableOpReadVariableOp%head_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_2/MatMul/ReadVariableOp�
head_2/MatMulMatMuldens_2/Relu:activations:0$head_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_2/MatMul�
head_2/BiasAdd/ReadVariableOpReadVariableOp&head_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_2/BiasAdd/ReadVariableOp�
head_2/BiasAddBiasAddhead_2/MatMul:product:0%head_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_2/BiasAddv
head_2/SigmoidSigmoidhead_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_2/Sigmoid�
head_1/MatMul/ReadVariableOpReadVariableOp%head_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_1/MatMul/ReadVariableOp�
head_1/MatMulMatMuldens_1/Relu:activations:0$head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_1/MatMul�
head_1/BiasAdd/ReadVariableOpReadVariableOp&head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_1/BiasAdd/ReadVariableOp�
head_1/BiasAddBiasAddhead_1/MatMul:product:0%head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_1/BiasAddv
head_1/SigmoidSigmoidhead_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_1/Sigmoid�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentityhead_1/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identityhead_2/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identityhead_3/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identityhead_4/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identityhead_5/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identityhead_6/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identityhead_7/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp2>
dens_1/BiasAdd/ReadVariableOpdens_1/BiasAdd/ReadVariableOp2<
dens_1/MatMul/ReadVariableOpdens_1/MatMul/ReadVariableOp2>
dens_2/BiasAdd/ReadVariableOpdens_2/BiasAdd/ReadVariableOp2<
dens_2/MatMul/ReadVariableOpdens_2/MatMul/ReadVariableOp2>
dens_3/BiasAdd/ReadVariableOpdens_3/BiasAdd/ReadVariableOp2<
dens_3/MatMul/ReadVariableOpdens_3/MatMul/ReadVariableOp2>
dens_4/BiasAdd/ReadVariableOpdens_4/BiasAdd/ReadVariableOp2<
dens_4/MatMul/ReadVariableOpdens_4/MatMul/ReadVariableOp2>
dens_5/BiasAdd/ReadVariableOpdens_5/BiasAdd/ReadVariableOp2<
dens_5/MatMul/ReadVariableOpdens_5/MatMul/ReadVariableOp2>
dens_6/BiasAdd/ReadVariableOpdens_6/BiasAdd/ReadVariableOp2<
dens_6/MatMul/ReadVariableOpdens_6/MatMul/ReadVariableOp2>
dens_7/BiasAdd/ReadVariableOpdens_7/BiasAdd/ReadVariableOp2<
dens_7/MatMul/ReadVariableOpdens_7/MatMul/ReadVariableOp2>
head_1/BiasAdd/ReadVariableOphead_1/BiasAdd/ReadVariableOp2<
head_1/MatMul/ReadVariableOphead_1/MatMul/ReadVariableOp2>
head_2/BiasAdd/ReadVariableOphead_2/BiasAdd/ReadVariableOp2<
head_2/MatMul/ReadVariableOphead_2/MatMul/ReadVariableOp2>
head_3/BiasAdd/ReadVariableOphead_3/BiasAdd/ReadVariableOp2<
head_3/MatMul/ReadVariableOphead_3/MatMul/ReadVariableOp2>
head_4/BiasAdd/ReadVariableOphead_4/BiasAdd/ReadVariableOp2<
head_4/MatMul/ReadVariableOphead_4/MatMul/ReadVariableOp2>
head_5/BiasAdd/ReadVariableOphead_5/BiasAdd/ReadVariableOp2<
head_5/MatMul/ReadVariableOphead_5/MatMul/ReadVariableOp2>
head_6/BiasAdd/ReadVariableOphead_6/BiasAdd/ReadVariableOp2<
head_6/MatMul/ReadVariableOphead_6/MatMul/ReadVariableOp2>
head_7/BiasAdd/ReadVariableOphead_7/BiasAdd/ReadVariableOp2<
head_7/MatMul/ReadVariableOphead_7/MatMul/ReadVariableOp2L
$trunk_/conv_1/BiasAdd/ReadVariableOp$trunk_/conv_1/BiasAdd/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_1/ReadVariableOp&trunk_/conv_1/BiasAdd_1/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_2/ReadVariableOp&trunk_/conv_1/BiasAdd_2/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_3/ReadVariableOp&trunk_/conv_1/BiasAdd_3/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_4/ReadVariableOp&trunk_/conv_1/BiasAdd_4/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_5/ReadVariableOp&trunk_/conv_1/BiasAdd_5/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_6/ReadVariableOp&trunk_/conv_1/BiasAdd_6/ReadVariableOp2d
0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp2L
$trunk_/conv_2/BiasAdd/ReadVariableOp$trunk_/conv_2/BiasAdd/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_1/ReadVariableOp&trunk_/conv_2/BiasAdd_1/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_2/ReadVariableOp&trunk_/conv_2/BiasAdd_2/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_3/ReadVariableOp&trunk_/conv_2/BiasAdd_3/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_4/ReadVariableOp&trunk_/conv_2/BiasAdd_4/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_5/ReadVariableOp&trunk_/conv_2/BiasAdd_5/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_6/ReadVariableOp&trunk_/conv_2/BiasAdd_6/ReadVariableOp2d
0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp2L
$trunk_/conv_3/BiasAdd/ReadVariableOp$trunk_/conv_3/BiasAdd/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_1/ReadVariableOp&trunk_/conv_3/BiasAdd_1/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_2/ReadVariableOp&trunk_/conv_3/BiasAdd_2/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_3/ReadVariableOp&trunk_/conv_3/BiasAdd_3/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_4/ReadVariableOp&trunk_/conv_3/BiasAdd_4/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_5/ReadVariableOp&trunk_/conv_3/BiasAdd_5/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_6/ReadVariableOp&trunk_/conv_3/BiasAdd_6/ReadVariableOp2d
0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/2:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/3:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/4:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/5:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/6
�#
�	
;__inference_multi-task_self-supervised_layer_call_fn_218532
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@`
	unknown_4:`
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:

unknown_23:	�

unknown_24:

unknown_25:	�

unknown_26:

unknown_27:	�

unknown_28:

unknown_29:	�

unknown_30:

unknown_31:	�

unknown_32:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*4
Tin-
+2)*
Tout
	2*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������*D
_read_only_resource_inputs&
$"	
 !"#$%&'(*2
config_proto" 

CPU

GPU2*0,1J 8� *_
fZRX
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_2177272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/2:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/3:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/4:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/5:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/6
�
�
'__inference_dens_6_layer_call_fn_219835

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_6_layer_call_and_return_conditional_losses_2170042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_220231N
8conv_3_kernel_regularizer_square_readvariableop_resource:@`
identity��/conv_3/kernel/Regularizer/Square/ReadVariableOp�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv_3_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentity!conv_3/kernel/Regularizer/mul:z:00^conv_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp
�

�
B__inference_head_6_layer_call_and_return_conditional_losses_219986

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dens_7_layer_call_and_return_conditional_losses_219866

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_conv_1_layer_call_fn_220021

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_2164342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������i 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
`
B__inference_drop_1_layer_call_and_return_conditional_losses_220058

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:���������i 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:���������i 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������i :S O
+
_output_shapes
:���������i 
 
_user_specified_nameinputs
�7
�
B__inference_trunk__layer_call_and_return_conditional_losses_216536

inputs#
conv_1_216435: 
conv_1_216437: #
conv_2_216470: @
conv_2_216472:@#
conv_3_216505:@`
conv_3_216507:`
identity��conv_1/StatefulPartitionedCall�/conv_1/kernel/Regularizer/Square/ReadVariableOp�conv_2/StatefulPartitionedCall�/conv_2/kernel/Regularizer/Square/ReadVariableOp�conv_3/StatefulPartitionedCall�/conv_3/kernel/Regularizer/Square/ReadVariableOp�
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_216435conv_1_216437*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_2164342 
conv_1/StatefulPartitionedCall�
drop_1/PartitionedCallPartitionedCall'conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������i * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_1_layer_call_and_return_conditional_losses_2164452
drop_1/PartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCalldrop_1/PartitionedCall:output:0conv_2_216470conv_2_216472*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_2164692 
conv_2/StatefulPartitionedCall�
drop_2/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Z@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_2_layer_call_and_return_conditional_losses_2164802
drop_2/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCalldrop_2/PartitionedCall:output:0conv_3_216505conv_3_216507*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_2165042 
conv_3/StatefulPartitionedCall�
drop_3/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_drop_3_layer_call_and_return_conditional_losses_2165152
drop_3/PartitionedCall�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_1_216435*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_2_216470*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv_3_216505*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentitydrop_3/PartitionedCall:output:0^conv_1/StatefulPartitionedCall0^conv_1/kernel/Regularizer/Square/ReadVariableOp^conv_2/StatefulPartitionedCall0^conv_2/kernel/Regularizer/Square/ReadVariableOp^conv_3/StatefulPartitionedCall0^conv_3/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dens_3_layer_call_and_return_conditional_losses_219786

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_conv_1_layer_call_and_return_conditional_losses_220043

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�/conv_1/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������i 2
Relu�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp0^conv_1/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������i 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dens_2_layer_call_and_return_conditional_losses_219766

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dens_7_layer_call_fn_219855

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_7_layer_call_and_return_conditional_losses_2169872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�O
"__inference__traced_restore_221109
file_prefix2
assignvariableop_dens_1_kernel:
��-
assignvariableop_1_dens_1_bias:	�4
 assignvariableop_2_dens_2_kernel:
��-
assignvariableop_3_dens_2_bias:	�4
 assignvariableop_4_dens_3_kernel:
��-
assignvariableop_5_dens_3_bias:	�4
 assignvariableop_6_dens_4_kernel:
��-
assignvariableop_7_dens_4_bias:	�4
 assignvariableop_8_dens_5_kernel:
��-
assignvariableop_9_dens_5_bias:	�5
!assignvariableop_10_dens_6_kernel:
��.
assignvariableop_11_dens_6_bias:	�5
!assignvariableop_12_dens_7_kernel:
��.
assignvariableop_13_dens_7_bias:	�4
!assignvariableop_14_head_1_kernel:	�-
assignvariableop_15_head_1_bias:4
!assignvariableop_16_head_2_kernel:	�-
assignvariableop_17_head_2_bias:4
!assignvariableop_18_head_3_kernel:	�-
assignvariableop_19_head_3_bias:4
!assignvariableop_20_head_4_kernel:	�-
assignvariableop_21_head_4_bias:4
!assignvariableop_22_head_5_kernel:	�-
assignvariableop_23_head_5_bias:4
!assignvariableop_24_head_6_kernel:	�-
assignvariableop_25_head_6_bias:4
!assignvariableop_26_head_7_kernel:	�-
assignvariableop_27_head_7_bias:'
assignvariableop_28_adam_iter:	 )
assignvariableop_29_adam_beta_1: )
assignvariableop_30_adam_beta_2: (
assignvariableop_31_adam_decay: 0
&assignvariableop_32_adam_learning_rate: 7
!assignvariableop_33_conv_1_kernel: -
assignvariableop_34_conv_1_bias: 7
!assignvariableop_35_conv_2_kernel: @-
assignvariableop_36_conv_2_bias:@7
!assignvariableop_37_conv_3_kernel:@`-
assignvariableop_38_conv_3_bias:`#
assignvariableop_39_total: #
assignvariableop_40_count: %
assignvariableop_41_total_1: %
assignvariableop_42_count_1: %
assignvariableop_43_total_2: %
assignvariableop_44_count_2: %
assignvariableop_45_total_3: %
assignvariableop_46_count_3: %
assignvariableop_47_total_4: %
assignvariableop_48_count_4: %
assignvariableop_49_total_5: %
assignvariableop_50_count_5: %
assignvariableop_51_total_6: %
assignvariableop_52_count_6: %
assignvariableop_53_total_7: %
assignvariableop_54_count_7: %
assignvariableop_55_total_8: %
assignvariableop_56_count_8: %
assignvariableop_57_total_9: %
assignvariableop_58_count_9: &
assignvariableop_59_total_10: &
assignvariableop_60_count_10: &
assignvariableop_61_total_11: &
assignvariableop_62_count_11: &
assignvariableop_63_total_12: &
assignvariableop_64_count_12: &
assignvariableop_65_total_13: &
assignvariableop_66_count_13: &
assignvariableop_67_total_14: &
assignvariableop_68_count_14: <
(assignvariableop_69_adam_dens_1_kernel_m:
��5
&assignvariableop_70_adam_dens_1_bias_m:	�<
(assignvariableop_71_adam_dens_2_kernel_m:
��5
&assignvariableop_72_adam_dens_2_bias_m:	�<
(assignvariableop_73_adam_dens_3_kernel_m:
��5
&assignvariableop_74_adam_dens_3_bias_m:	�<
(assignvariableop_75_adam_dens_4_kernel_m:
��5
&assignvariableop_76_adam_dens_4_bias_m:	�<
(assignvariableop_77_adam_dens_5_kernel_m:
��5
&assignvariableop_78_adam_dens_5_bias_m:	�<
(assignvariableop_79_adam_dens_6_kernel_m:
��5
&assignvariableop_80_adam_dens_6_bias_m:	�<
(assignvariableop_81_adam_dens_7_kernel_m:
��5
&assignvariableop_82_adam_dens_7_bias_m:	�;
(assignvariableop_83_adam_head_1_kernel_m:	�4
&assignvariableop_84_adam_head_1_bias_m:;
(assignvariableop_85_adam_head_2_kernel_m:	�4
&assignvariableop_86_adam_head_2_bias_m:;
(assignvariableop_87_adam_head_3_kernel_m:	�4
&assignvariableop_88_adam_head_3_bias_m:;
(assignvariableop_89_adam_head_4_kernel_m:	�4
&assignvariableop_90_adam_head_4_bias_m:;
(assignvariableop_91_adam_head_5_kernel_m:	�4
&assignvariableop_92_adam_head_5_bias_m:;
(assignvariableop_93_adam_head_6_kernel_m:	�4
&assignvariableop_94_adam_head_6_bias_m:;
(assignvariableop_95_adam_head_7_kernel_m:	�4
&assignvariableop_96_adam_head_7_bias_m:>
(assignvariableop_97_adam_conv_1_kernel_m: 4
&assignvariableop_98_adam_conv_1_bias_m: >
(assignvariableop_99_adam_conv_2_kernel_m: @5
'assignvariableop_100_adam_conv_2_bias_m:@?
)assignvariableop_101_adam_conv_3_kernel_m:@`5
'assignvariableop_102_adam_conv_3_bias_m:`=
)assignvariableop_103_adam_dens_1_kernel_v:
��6
'assignvariableop_104_adam_dens_1_bias_v:	�=
)assignvariableop_105_adam_dens_2_kernel_v:
��6
'assignvariableop_106_adam_dens_2_bias_v:	�=
)assignvariableop_107_adam_dens_3_kernel_v:
��6
'assignvariableop_108_adam_dens_3_bias_v:	�=
)assignvariableop_109_adam_dens_4_kernel_v:
��6
'assignvariableop_110_adam_dens_4_bias_v:	�=
)assignvariableop_111_adam_dens_5_kernel_v:
��6
'assignvariableop_112_adam_dens_5_bias_v:	�=
)assignvariableop_113_adam_dens_6_kernel_v:
��6
'assignvariableop_114_adam_dens_6_bias_v:	�=
)assignvariableop_115_adam_dens_7_kernel_v:
��6
'assignvariableop_116_adam_dens_7_bias_v:	�<
)assignvariableop_117_adam_head_1_kernel_v:	�5
'assignvariableop_118_adam_head_1_bias_v:<
)assignvariableop_119_adam_head_2_kernel_v:	�5
'assignvariableop_120_adam_head_2_bias_v:<
)assignvariableop_121_adam_head_3_kernel_v:	�5
'assignvariableop_122_adam_head_3_bias_v:<
)assignvariableop_123_adam_head_4_kernel_v:	�5
'assignvariableop_124_adam_head_4_bias_v:<
)assignvariableop_125_adam_head_5_kernel_v:	�5
'assignvariableop_126_adam_head_5_bias_v:<
)assignvariableop_127_adam_head_6_kernel_v:	�5
'assignvariableop_128_adam_head_6_bias_v:<
)assignvariableop_129_adam_head_7_kernel_v:	�5
'assignvariableop_130_adam_head_7_bias_v:?
)assignvariableop_131_adam_conv_1_kernel_v: 5
'assignvariableop_132_adam_conv_1_bias_v: ?
)assignvariableop_133_adam_conv_2_kernel_v: @5
'assignvariableop_134_adam_conv_2_bias_v:@?
)assignvariableop_135_adam_conv_3_kernel_v:@`5
'assignvariableop_136_adam_conv_3_bias_v:`
identity_138��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�H
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�G
value�GB�G�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dens_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dens_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_dens_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dens_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_dens_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dens_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_dens_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dens_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp assignvariableop_8_dens_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dens_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dens_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_dens_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dens_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_dens_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_head_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_head_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp!assignvariableop_16_head_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_head_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp!assignvariableop_18_head_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_head_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp!assignvariableop_20_head_4_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_head_4_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp!assignvariableop_22_head_5_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_head_5_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp!assignvariableop_24_head_6_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_head_6_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp!assignvariableop_26_head_7_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_head_7_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_iterIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_beta_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_beta_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_decayIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_learning_rateIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp!assignvariableop_33_conv_1_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOpassignvariableop_34_conv_1_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp!assignvariableop_35_conv_2_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOpassignvariableop_36_conv_2_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv_3_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpassignvariableop_38_conv_3_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOpassignvariableop_39_totalIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOpassignvariableop_40_countIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_2Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_2Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_3Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_3Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_4Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_4Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_5Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_5Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOpassignvariableop_51_total_6Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_6Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_7Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_7Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOpassignvariableop_55_total_8Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOpassignvariableop_56_count_8Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOpassignvariableop_57_total_9Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOpassignvariableop_58_count_9Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOpassignvariableop_59_total_10Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOpassignvariableop_60_count_10Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOpassignvariableop_61_total_11Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOpassignvariableop_62_count_11Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOpassignvariableop_63_total_12Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOpassignvariableop_64_count_12Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOpassignvariableop_65_total_13Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOpassignvariableop_66_count_13Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOpassignvariableop_67_total_14Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOpassignvariableop_68_count_14Identity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_dens_1_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp&assignvariableop_70_adam_dens_1_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_dens_2_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp&assignvariableop_72_adam_dens_2_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_dens_3_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp&assignvariableop_74_adam_dens_3_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp(assignvariableop_75_adam_dens_4_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp&assignvariableop_76_adam_dens_4_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adam_dens_5_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp&assignvariableop_78_adam_dens_5_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adam_dens_6_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp&assignvariableop_80_adam_dens_6_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp(assignvariableop_81_adam_dens_7_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp&assignvariableop_82_adam_dens_7_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83�
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_head_1_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84�
AssignVariableOp_84AssignVariableOp&assignvariableop_84_adam_head_1_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85�
AssignVariableOp_85AssignVariableOp(assignvariableop_85_adam_head_2_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86�
AssignVariableOp_86AssignVariableOp&assignvariableop_86_adam_head_2_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87�
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_head_3_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88�
AssignVariableOp_88AssignVariableOp&assignvariableop_88_adam_head_3_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89�
AssignVariableOp_89AssignVariableOp(assignvariableop_89_adam_head_4_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90�
AssignVariableOp_90AssignVariableOp&assignvariableop_90_adam_head_4_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91�
AssignVariableOp_91AssignVariableOp(assignvariableop_91_adam_head_5_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92�
AssignVariableOp_92AssignVariableOp&assignvariableop_92_adam_head_5_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93�
AssignVariableOp_93AssignVariableOp(assignvariableop_93_adam_head_6_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94�
AssignVariableOp_94AssignVariableOp&assignvariableop_94_adam_head_6_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95�
AssignVariableOp_95AssignVariableOp(assignvariableop_95_adam_head_7_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96�
AssignVariableOp_96AssignVariableOp&assignvariableop_96_adam_head_7_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97�
AssignVariableOp_97AssignVariableOp(assignvariableop_97_adam_conv_1_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98�
AssignVariableOp_98AssignVariableOp&assignvariableop_98_adam_conv_1_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99�
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_conv_2_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100�
AssignVariableOp_100AssignVariableOp'assignvariableop_100_adam_conv_2_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101�
AssignVariableOp_101AssignVariableOp)assignvariableop_101_adam_conv_3_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102�
AssignVariableOp_102AssignVariableOp'assignvariableop_102_adam_conv_3_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103�
AssignVariableOp_103AssignVariableOp)assignvariableop_103_adam_dens_1_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104�
AssignVariableOp_104AssignVariableOp'assignvariableop_104_adam_dens_1_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105�
AssignVariableOp_105AssignVariableOp)assignvariableop_105_adam_dens_2_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106�
AssignVariableOp_106AssignVariableOp'assignvariableop_106_adam_dens_2_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107�
AssignVariableOp_107AssignVariableOp)assignvariableop_107_adam_dens_3_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108�
AssignVariableOp_108AssignVariableOp'assignvariableop_108_adam_dens_3_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109�
AssignVariableOp_109AssignVariableOp)assignvariableop_109_adam_dens_4_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110�
AssignVariableOp_110AssignVariableOp'assignvariableop_110_adam_dens_4_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111�
AssignVariableOp_111AssignVariableOp)assignvariableop_111_adam_dens_5_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112�
AssignVariableOp_112AssignVariableOp'assignvariableop_112_adam_dens_5_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113�
AssignVariableOp_113AssignVariableOp)assignvariableop_113_adam_dens_6_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114�
AssignVariableOp_114AssignVariableOp'assignvariableop_114_adam_dens_6_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115�
AssignVariableOp_115AssignVariableOp)assignvariableop_115_adam_dens_7_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116�
AssignVariableOp_116AssignVariableOp'assignvariableop_116_adam_dens_7_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117�
AssignVariableOp_117AssignVariableOp)assignvariableop_117_adam_head_1_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118�
AssignVariableOp_118AssignVariableOp'assignvariableop_118_adam_head_1_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119�
AssignVariableOp_119AssignVariableOp)assignvariableop_119_adam_head_2_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120�
AssignVariableOp_120AssignVariableOp'assignvariableop_120_adam_head_2_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121�
AssignVariableOp_121AssignVariableOp)assignvariableop_121_adam_head_3_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122�
AssignVariableOp_122AssignVariableOp'assignvariableop_122_adam_head_3_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123�
AssignVariableOp_123AssignVariableOp)assignvariableop_123_adam_head_4_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124�
AssignVariableOp_124AssignVariableOp'assignvariableop_124_adam_head_4_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125�
AssignVariableOp_125AssignVariableOp)assignvariableop_125_adam_head_5_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126�
AssignVariableOp_126AssignVariableOp'assignvariableop_126_adam_head_5_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127�
AssignVariableOp_127AssignVariableOp)assignvariableop_127_adam_head_6_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128�
AssignVariableOp_128AssignVariableOp'assignvariableop_128_adam_head_6_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129�
AssignVariableOp_129AssignVariableOp)assignvariableop_129_adam_head_7_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130�
AssignVariableOp_130AssignVariableOp'assignvariableop_130_adam_head_7_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131�
AssignVariableOp_131AssignVariableOp)assignvariableop_131_adam_conv_1_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132�
AssignVariableOp_132AssignVariableOp'assignvariableop_132_adam_conv_1_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133�
AssignVariableOp_133AssignVariableOp)assignvariableop_133_adam_conv_2_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134�
AssignVariableOp_134AssignVariableOp'assignvariableop_134_adam_conv_2_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135�
AssignVariableOp_135AssignVariableOp)assignvariableop_135_adam_conv_3_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136�
AssignVariableOp_136AssignVariableOp'assignvariableop_136_adam_conv_3_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_137Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_137�
Identity_138IdentityIdentity_137:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_138"%
identity_138Identity_138:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�"
�	
$__inference_signature_wrapper_218350
input_1
input_2
input_3
input_4
input_5
input_6
input_7
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@`
	unknown_4:`
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:
��

unknown_14:	�

unknown_15:
��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:	�

unknown_20:

unknown_21:	�

unknown_22:

unknown_23:	�

unknown_24:

unknown_25:	�

unknown_26:

unknown_27:	�

unknown_28:

unknown_29:	�

unknown_30:

unknown_31:	�

unknown_32:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6input_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*4
Tin-
+2)*
Tout
	2*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:���������:���������:���������:���������:���������*D
_read_only_resource_inputs&
$"	
 !"#$%&'(*2
config_proto" 

CPU

GPU2*0,1J 8� **
f%R#
!__inference__wrapped_model_2164052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identity StatefulPartitionedCall:output:4^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identity StatefulPartitionedCall:output:5^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identity StatefulPartitionedCall:output:6^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:����������
!
_user_specified_name	input_1:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_2:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_3:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_4:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_5:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_6:UQ
,
_output_shapes
:����������
!
_user_specified_name	input_7
�V
�
B__inference_trunk__layer_call_and_return_conditional_losses_219614

inputsH
2conv_1_conv1d_expanddims_1_readvariableop_resource: 4
&conv_1_biasadd_readvariableop_resource: H
2conv_2_conv1d_expanddims_1_readvariableop_resource: @4
&conv_2_biasadd_readvariableop_resource:@H
2conv_3_conv1d_expanddims_1_readvariableop_resource:@`4
&conv_3_biasadd_readvariableop_resource:`
identity��conv_1/BiasAdd/ReadVariableOp�)conv_1/conv1d/ExpandDims_1/ReadVariableOp�/conv_1/kernel/Regularizer/Square/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�)conv_2/conv1d/ExpandDims_1/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�)conv_3/conv1d/ExpandDims_1/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOp�
conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_1/conv1d/ExpandDims/dim�
conv_1/conv1d/ExpandDims
ExpandDimsinputs%conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2
conv_1/conv1d/ExpandDims�
)conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02+
)conv_1/conv1d/ExpandDims_1/ReadVariableOp�
conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_1/conv1d/ExpandDims_1/dim�
conv_1/conv1d/ExpandDims_1
ExpandDims1conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv_1/conv1d/ExpandDims_1�
conv_1/conv1dConv2D!conv_1/conv1d/ExpandDims:output:0#conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
conv_1/conv1d�
conv_1/conv1d/SqueezeSqueezeconv_1/conv1d:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2
conv_1/conv1d/Squeeze�
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_1/BiasAdd/ReadVariableOp�
conv_1/BiasAddBiasAddconv_1/conv1d/Squeeze:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
conv_1/BiasAddq
conv_1/ReluReluconv_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������i 2
conv_1/Relu
drop_1/IdentityIdentityconv_1/Relu:activations:0*
T0*+
_output_shapes
:���������i 2
drop_1/Identity�
conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_2/conv1d/ExpandDims/dim�
conv_2/conv1d/ExpandDims
ExpandDimsdrop_1/Identity:output:0%conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2
conv_2/conv1d/ExpandDims�
)conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02+
)conv_2/conv1d/ExpandDims_1/ReadVariableOp�
conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_2/conv1d/ExpandDims_1/dim�
conv_2/conv1d/ExpandDims_1
ExpandDims1conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv_2/conv1d/ExpandDims_1�
conv_2/conv1dConv2D!conv_2/conv1d/ExpandDims:output:0#conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
conv_2/conv1d�
conv_2/conv1d/SqueezeSqueezeconv_2/conv1d:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2
conv_2/conv1d/Squeeze�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_2/BiasAdd/ReadVariableOp�
conv_2/BiasAddBiasAddconv_2/conv1d/Squeeze:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
conv_2/BiasAddq
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������Z@2
conv_2/Relu
drop_2/IdentityIdentityconv_2/Relu:activations:0*
T0*+
_output_shapes
:���������Z@2
drop_2/Identity�
conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv_3/conv1d/ExpandDims/dim�
conv_3/conv1d/ExpandDims
ExpandDimsdrop_2/Identity:output:0%conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2
conv_3/conv1d/ExpandDims�
)conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02+
)conv_3/conv1d/ExpandDims_1/ReadVariableOp�
conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv_3/conv1d/ExpandDims_1/dim�
conv_3/conv1d/ExpandDims_1
ExpandDims1conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2
conv_3/conv1d/ExpandDims_1�
conv_3/conv1dConv2D!conv_3/conv1d/ExpandDims:output:0#conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
conv_3/conv1d�
conv_3/conv1d/SqueezeSqueezeconv_3/conv1d:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2
conv_3/conv1d/Squeeze�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/conv1d/Squeeze:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
conv_3/BiasAddq
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������S`2
conv_3/Relu
drop_3/IdentityIdentityconv_3/Relu:activations:0*
T0*+
_output_shapes
:���������S`2
drop_3/Identity�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentitydrop_3/Identity:output:0^conv_1/BiasAdd/ReadVariableOp*^conv_1/conv1d/ExpandDims_1/ReadVariableOp0^conv_1/kernel/Regularizer/Square/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp*^conv_2/conv1d/ExpandDims_1/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp*^conv_3/conv1d/ExpandDims_1/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2V
)conv_1/conv1d/ExpandDims_1/ReadVariableOp)conv_1/conv1d/ExpandDims_1/ReadVariableOp2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2V
)conv_2/conv1d/ExpandDims_1/ReadVariableOp)conv_2/conv1d/ExpandDims_1/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2V
)conv_3/conv1d/ExpandDims_1/ReadVariableOp)conv_3/conv1d/ExpandDims_1/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
h
L__inference_global_max_pool__layer_call_and_return_conditional_losses_219726

inputs
identityn
pool_/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pool_/ExpandDims/dim�
pool_/ExpandDims
ExpandDimsinputspool_/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������S`2
pool_/ExpandDims�
pool_/MaxPoolMaxPoolpool_/ExpandDims:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2
pool_/MaxPool�
pool_/SqueezeSqueezepool_/MaxPool:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2
pool_/Squeezek
flat_/ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
flat_/Const�
flat_/ReshapeReshapepool_/Squeeze:output:0flat_/Const:output:0*
T0*(
_output_shapes
:����������2
flat_/Reshapek
IdentityIdentityflat_/Reshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�
�
'__inference_dens_4_layer_call_fn_219795

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dens_4_layer_call_and_return_conditional_losses_2170382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
h
L__inference_global_max_pool__layer_call_and_return_conditional_losses_216875

inputs
identity�
pool_/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������)`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_pool__layer_call_and_return_conditional_losses_2168302
pool_/PartitionedCall�
flat_/PartitionedCallPartitionedCallpool_/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_flat__layer_call_and_return_conditional_losses_2168502
flat_/PartitionedCalls
IdentityIdentityflat_/PartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�

�
B__inference_head_3_layer_call_and_return_conditional_losses_217174

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
B
&__inference_flat__layer_call_fn_220236

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *J
fERC
A__inference_flat__layer_call_and_return_conditional_losses_2168502
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������)`:S O
+
_output_shapes
:���������)`
 
_user_specified_nameinputs
�
�
'__inference_conv_3_layer_call_fn_220149

inputs
unknown:@`
	unknown_0:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_2165042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Z@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Z@
 
_user_specified_nameinputs
�
�
'__inference_trunk__layer_call_fn_219536

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:@`
	unknown_4:`
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������S`*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_trunk__layer_call_and_return_conditional_losses_2165362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_dens_4_layer_call_and_return_conditional_losses_219806

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
h
L__inference_global_max_pool__layer_call_and_return_conditional_losses_219716

inputs
identityn
pool_/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
pool_/ExpandDims/dim�
pool_/ExpandDims
ExpandDimsinputspool_/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������S`2
pool_/ExpandDims�
pool_/MaxPoolMaxPoolpool_/ExpandDims:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2
pool_/MaxPool�
pool_/SqueezeSqueezepool_/MaxPool:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2
pool_/Squeezek
flat_/ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
flat_/Const�
flat_/ReshapeReshapepool_/Squeeze:output:0flat_/Const:output:0*
T0*(
_output_shapes
:����������2
flat_/Reshapek
IdentityIdentityflat_/Reshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������S`:S O
+
_output_shapes
:���������S`
 
_user_specified_nameinputs
�

�
B__inference_dens_2_layer_call_and_return_conditional_losses_217072

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
B__inference_head_3_layer_call_and_return_conditional_losses_219926

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_conv_3_layer_call_and_return_conditional_losses_220171

inputsA
+conv1d_expanddims_1_readvariableop_resource:@`-
biasadd_readvariableop_resource:`
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������S`2
Relu�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp*
T0*+
_output_shapes
:���������S`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Z@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:���������Z@
 
_user_specified_nameinputs
��
�)
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_218943
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6O
9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource: ;
-trunk__conv_1_biasadd_readvariableop_resource: O
9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource: @;
-trunk__conv_2_biasadd_readvariableop_resource:@O
9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource:@`;
-trunk__conv_3_biasadd_readvariableop_resource:`9
%dens_7_matmul_readvariableop_resource:
��5
&dens_7_biasadd_readvariableop_resource:	�9
%dens_6_matmul_readvariableop_resource:
��5
&dens_6_biasadd_readvariableop_resource:	�9
%dens_5_matmul_readvariableop_resource:
��5
&dens_5_biasadd_readvariableop_resource:	�9
%dens_4_matmul_readvariableop_resource:
��5
&dens_4_biasadd_readvariableop_resource:	�9
%dens_3_matmul_readvariableop_resource:
��5
&dens_3_biasadd_readvariableop_resource:	�9
%dens_2_matmul_readvariableop_resource:
��5
&dens_2_biasadd_readvariableop_resource:	�9
%dens_1_matmul_readvariableop_resource:
��5
&dens_1_biasadd_readvariableop_resource:	�8
%head_7_matmul_readvariableop_resource:	�4
&head_7_biasadd_readvariableop_resource:8
%head_6_matmul_readvariableop_resource:	�4
&head_6_biasadd_readvariableop_resource:8
%head_5_matmul_readvariableop_resource:	�4
&head_5_biasadd_readvariableop_resource:8
%head_4_matmul_readvariableop_resource:	�4
&head_4_biasadd_readvariableop_resource:8
%head_3_matmul_readvariableop_resource:	�4
&head_3_biasadd_readvariableop_resource:8
%head_2_matmul_readvariableop_resource:	�4
&head_2_biasadd_readvariableop_resource:8
%head_1_matmul_readvariableop_resource:	�4
&head_1_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6��/conv_1/kernel/Regularizer/Square/ReadVariableOp�/conv_2/kernel/Regularizer/Square/ReadVariableOp�/conv_3/kernel/Regularizer/Square/ReadVariableOp�dens_1/BiasAdd/ReadVariableOp�dens_1/MatMul/ReadVariableOp�dens_2/BiasAdd/ReadVariableOp�dens_2/MatMul/ReadVariableOp�dens_3/BiasAdd/ReadVariableOp�dens_3/MatMul/ReadVariableOp�dens_4/BiasAdd/ReadVariableOp�dens_4/MatMul/ReadVariableOp�dens_5/BiasAdd/ReadVariableOp�dens_5/MatMul/ReadVariableOp�dens_6/BiasAdd/ReadVariableOp�dens_6/MatMul/ReadVariableOp�dens_7/BiasAdd/ReadVariableOp�dens_7/MatMul/ReadVariableOp�head_1/BiasAdd/ReadVariableOp�head_1/MatMul/ReadVariableOp�head_2/BiasAdd/ReadVariableOp�head_2/MatMul/ReadVariableOp�head_3/BiasAdd/ReadVariableOp�head_3/MatMul/ReadVariableOp�head_4/BiasAdd/ReadVariableOp�head_4/MatMul/ReadVariableOp�head_5/BiasAdd/ReadVariableOp�head_5/MatMul/ReadVariableOp�head_6/BiasAdd/ReadVariableOp�head_6/MatMul/ReadVariableOp�head_7/BiasAdd/ReadVariableOp�head_7/MatMul/ReadVariableOp�$trunk_/conv_1/BiasAdd/ReadVariableOp�&trunk_/conv_1/BiasAdd_1/ReadVariableOp�&trunk_/conv_1/BiasAdd_2/ReadVariableOp�&trunk_/conv_1/BiasAdd_3/ReadVariableOp�&trunk_/conv_1/BiasAdd_4/ReadVariableOp�&trunk_/conv_1/BiasAdd_5/ReadVariableOp�&trunk_/conv_1/BiasAdd_6/ReadVariableOp�0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp�2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp�$trunk_/conv_2/BiasAdd/ReadVariableOp�&trunk_/conv_2/BiasAdd_1/ReadVariableOp�&trunk_/conv_2/BiasAdd_2/ReadVariableOp�&trunk_/conv_2/BiasAdd_3/ReadVariableOp�&trunk_/conv_2/BiasAdd_4/ReadVariableOp�&trunk_/conv_2/BiasAdd_5/ReadVariableOp�&trunk_/conv_2/BiasAdd_6/ReadVariableOp�0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp�2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp�$trunk_/conv_3/BiasAdd/ReadVariableOp�&trunk_/conv_3/BiasAdd_1/ReadVariableOp�&trunk_/conv_3/BiasAdd_2/ReadVariableOp�&trunk_/conv_3/BiasAdd_3/ReadVariableOp�&trunk_/conv_3/BiasAdd_4/ReadVariableOp�&trunk_/conv_3/BiasAdd_5/ReadVariableOp�&trunk_/conv_3/BiasAdd_6/ReadVariableOp�0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp�2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp�
#trunk_/conv_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#trunk_/conv_1/conv1d/ExpandDims/dim�
trunk_/conv_1/conv1d/ExpandDims
ExpandDimsinputs_6,trunk_/conv_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2!
trunk_/conv_1/conv1d/ExpandDims�
0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype022
0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp�
%trunk_/conv_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%trunk_/conv_1/conv1d/ExpandDims_1/dim�
!trunk_/conv_1/conv1d/ExpandDims_1
ExpandDims8trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp:value:0.trunk_/conv_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2#
!trunk_/conv_1/conv1d/ExpandDims_1�
trunk_/conv_1/conv1dConv2D(trunk_/conv_1/conv1d/ExpandDims:output:0*trunk_/conv_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d�
trunk_/conv_1/conv1d/SqueezeSqueezetrunk_/conv_1/conv1d:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2
trunk_/conv_1/conv1d/Squeeze�
$trunk_/conv_1/BiasAdd/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$trunk_/conv_1/BiasAdd/ReadVariableOp�
trunk_/conv_1/BiasAddBiasAdd%trunk_/conv_1/conv1d/Squeeze:output:0,trunk_/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd�
trunk_/conv_1/ReluRelutrunk_/conv_1/BiasAdd:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu�
trunk_/drop_1/IdentityIdentity trunk_/conv_1/Relu:activations:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/Identity�
#trunk_/conv_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#trunk_/conv_2/conv1d/ExpandDims/dim�
trunk_/conv_2/conv1d/ExpandDims
ExpandDimstrunk_/drop_1/Identity:output:0,trunk_/conv_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2!
trunk_/conv_2/conv1d/ExpandDims�
0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype022
0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp�
%trunk_/conv_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%trunk_/conv_2/conv1d/ExpandDims_1/dim�
!trunk_/conv_2/conv1d/ExpandDims_1
ExpandDims8trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp:value:0.trunk_/conv_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2#
!trunk_/conv_2/conv1d/ExpandDims_1�
trunk_/conv_2/conv1dConv2D(trunk_/conv_2/conv1d/ExpandDims:output:0*trunk_/conv_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d�
trunk_/conv_2/conv1d/SqueezeSqueezetrunk_/conv_2/conv1d:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2
trunk_/conv_2/conv1d/Squeeze�
$trunk_/conv_2/BiasAdd/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$trunk_/conv_2/BiasAdd/ReadVariableOp�
trunk_/conv_2/BiasAddBiasAdd%trunk_/conv_2/conv1d/Squeeze:output:0,trunk_/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd�
trunk_/conv_2/ReluRelutrunk_/conv_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu�
trunk_/drop_2/IdentityIdentity trunk_/conv_2/Relu:activations:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/Identity�
#trunk_/conv_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#trunk_/conv_3/conv1d/ExpandDims/dim�
trunk_/conv_3/conv1d/ExpandDims
ExpandDimstrunk_/drop_2/Identity:output:0,trunk_/conv_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2!
trunk_/conv_3/conv1d/ExpandDims�
0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype022
0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp�
%trunk_/conv_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2'
%trunk_/conv_3/conv1d/ExpandDims_1/dim�
!trunk_/conv_3/conv1d/ExpandDims_1
ExpandDims8trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp:value:0.trunk_/conv_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2#
!trunk_/conv_3/conv1d/ExpandDims_1�
trunk_/conv_3/conv1dConv2D(trunk_/conv_3/conv1d/ExpandDims:output:0*trunk_/conv_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d�
trunk_/conv_3/conv1d/SqueezeSqueezetrunk_/conv_3/conv1d:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2
trunk_/conv_3/conv1d/Squeeze�
$trunk_/conv_3/BiasAdd/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02&
$trunk_/conv_3/BiasAdd/ReadVariableOp�
trunk_/conv_3/BiasAddBiasAdd%trunk_/conv_3/conv1d/Squeeze:output:0,trunk_/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd�
trunk_/conv_3/ReluRelutrunk_/conv_3/BiasAdd:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu�
trunk_/drop_3/IdentityIdentity trunk_/conv_3/Relu:activations:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/Identity�
%trunk_/conv_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_1/ExpandDims/dim�
!trunk_/conv_1/conv1d_1/ExpandDims
ExpandDimsinputs_5.trunk_/conv_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_1/ExpandDims�
2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_1/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_1/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_1/ExpandDims_1�
trunk_/conv_1/conv1d_1Conv2D*trunk_/conv_1/conv1d_1/ExpandDims:output:0,trunk_/conv_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_1�
trunk_/conv_1/conv1d_1/SqueezeSqueezetrunk_/conv_1/conv1d_1:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_1/Squeeze�
&trunk_/conv_1/BiasAdd_1/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_1/ReadVariableOp�
trunk_/conv_1/BiasAdd_1BiasAdd'trunk_/conv_1/conv1d_1/Squeeze:output:0.trunk_/conv_1/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_1�
trunk_/conv_1/Relu_1Relu trunk_/conv_1/BiasAdd_1:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_1�
trunk_/drop_1/Identity_1Identity"trunk_/conv_1/Relu_1:activations:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/Identity_1�
%trunk_/conv_2/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_1/ExpandDims/dim�
!trunk_/conv_2/conv1d_1/ExpandDims
ExpandDims!trunk_/drop_1/Identity_1:output:0.trunk_/conv_2/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_1/ExpandDims�
2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_1/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_1/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_1/ExpandDims_1�
trunk_/conv_2/conv1d_1Conv2D*trunk_/conv_2/conv1d_1/ExpandDims:output:0,trunk_/conv_2/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_1�
trunk_/conv_2/conv1d_1/SqueezeSqueezetrunk_/conv_2/conv1d_1:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_1/Squeeze�
&trunk_/conv_2/BiasAdd_1/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_1/ReadVariableOp�
trunk_/conv_2/BiasAdd_1BiasAdd'trunk_/conv_2/conv1d_1/Squeeze:output:0.trunk_/conv_2/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_1�
trunk_/conv_2/Relu_1Relu trunk_/conv_2/BiasAdd_1:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_1�
trunk_/drop_2/Identity_1Identity"trunk_/conv_2/Relu_1:activations:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/Identity_1�
%trunk_/conv_3/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_1/ExpandDims/dim�
!trunk_/conv_3/conv1d_1/ExpandDims
ExpandDims!trunk_/drop_2/Identity_1:output:0.trunk_/conv_3/conv1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_1/ExpandDims�
2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_1/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_1/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_1/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_1/ExpandDims_1�
trunk_/conv_3/conv1d_1Conv2D*trunk_/conv_3/conv1d_1/ExpandDims:output:0,trunk_/conv_3/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_1�
trunk_/conv_3/conv1d_1/SqueezeSqueezetrunk_/conv_3/conv1d_1:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_1/Squeeze�
&trunk_/conv_3/BiasAdd_1/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_1/ReadVariableOp�
trunk_/conv_3/BiasAdd_1BiasAdd'trunk_/conv_3/conv1d_1/Squeeze:output:0.trunk_/conv_3/BiasAdd_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_1�
trunk_/conv_3/Relu_1Relu trunk_/conv_3/BiasAdd_1:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_1�
trunk_/drop_3/Identity_1Identity"trunk_/conv_3/Relu_1:activations:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/Identity_1�
%trunk_/conv_1/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_2/ExpandDims/dim�
!trunk_/conv_1/conv1d_2/ExpandDims
ExpandDimsinputs_4.trunk_/conv_1/conv1d_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_2/ExpandDims�
2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_2/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_2/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_2/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_2/ExpandDims_1�
trunk_/conv_1/conv1d_2Conv2D*trunk_/conv_1/conv1d_2/ExpandDims:output:0,trunk_/conv_1/conv1d_2/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_2�
trunk_/conv_1/conv1d_2/SqueezeSqueezetrunk_/conv_1/conv1d_2:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_2/Squeeze�
&trunk_/conv_1/BiasAdd_2/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_2/ReadVariableOp�
trunk_/conv_1/BiasAdd_2BiasAdd'trunk_/conv_1/conv1d_2/Squeeze:output:0.trunk_/conv_1/BiasAdd_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_2�
trunk_/conv_1/Relu_2Relu trunk_/conv_1/BiasAdd_2:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_2�
trunk_/drop_1/Identity_2Identity"trunk_/conv_1/Relu_2:activations:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/Identity_2�
%trunk_/conv_2/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_2/ExpandDims/dim�
!trunk_/conv_2/conv1d_2/ExpandDims
ExpandDims!trunk_/drop_1/Identity_2:output:0.trunk_/conv_2/conv1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_2/ExpandDims�
2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_2/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_2/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_2/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_2/ExpandDims_1�
trunk_/conv_2/conv1d_2Conv2D*trunk_/conv_2/conv1d_2/ExpandDims:output:0,trunk_/conv_2/conv1d_2/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_2�
trunk_/conv_2/conv1d_2/SqueezeSqueezetrunk_/conv_2/conv1d_2:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_2/Squeeze�
&trunk_/conv_2/BiasAdd_2/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_2/ReadVariableOp�
trunk_/conv_2/BiasAdd_2BiasAdd'trunk_/conv_2/conv1d_2/Squeeze:output:0.trunk_/conv_2/BiasAdd_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_2�
trunk_/conv_2/Relu_2Relu trunk_/conv_2/BiasAdd_2:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_2�
trunk_/drop_2/Identity_2Identity"trunk_/conv_2/Relu_2:activations:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/Identity_2�
%trunk_/conv_3/conv1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_2/ExpandDims/dim�
!trunk_/conv_3/conv1d_2/ExpandDims
ExpandDims!trunk_/drop_2/Identity_2:output:0.trunk_/conv_3/conv1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_2/ExpandDims�
2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_2/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_2/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_2/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_2/ExpandDims_1�
trunk_/conv_3/conv1d_2Conv2D*trunk_/conv_3/conv1d_2/ExpandDims:output:0,trunk_/conv_3/conv1d_2/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_2�
trunk_/conv_3/conv1d_2/SqueezeSqueezetrunk_/conv_3/conv1d_2:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_2/Squeeze�
&trunk_/conv_3/BiasAdd_2/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_2/ReadVariableOp�
trunk_/conv_3/BiasAdd_2BiasAdd'trunk_/conv_3/conv1d_2/Squeeze:output:0.trunk_/conv_3/BiasAdd_2/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_2�
trunk_/conv_3/Relu_2Relu trunk_/conv_3/BiasAdd_2:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_2�
trunk_/drop_3/Identity_2Identity"trunk_/conv_3/Relu_2:activations:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/Identity_2�
%trunk_/conv_1/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_3/ExpandDims/dim�
!trunk_/conv_1/conv1d_3/ExpandDims
ExpandDimsinputs_3.trunk_/conv_1/conv1d_3/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_3/ExpandDims�
2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_3/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_3/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_3/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_3/ExpandDims_1�
trunk_/conv_1/conv1d_3Conv2D*trunk_/conv_1/conv1d_3/ExpandDims:output:0,trunk_/conv_1/conv1d_3/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_3�
trunk_/conv_1/conv1d_3/SqueezeSqueezetrunk_/conv_1/conv1d_3:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_3/Squeeze�
&trunk_/conv_1/BiasAdd_3/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_3/ReadVariableOp�
trunk_/conv_1/BiasAdd_3BiasAdd'trunk_/conv_1/conv1d_3/Squeeze:output:0.trunk_/conv_1/BiasAdd_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_3�
trunk_/conv_1/Relu_3Relu trunk_/conv_1/BiasAdd_3:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_3�
trunk_/drop_1/Identity_3Identity"trunk_/conv_1/Relu_3:activations:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/Identity_3�
%trunk_/conv_2/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_3/ExpandDims/dim�
!trunk_/conv_2/conv1d_3/ExpandDims
ExpandDims!trunk_/drop_1/Identity_3:output:0.trunk_/conv_2/conv1d_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_3/ExpandDims�
2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_3/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_3/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_3/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_3/ExpandDims_1�
trunk_/conv_2/conv1d_3Conv2D*trunk_/conv_2/conv1d_3/ExpandDims:output:0,trunk_/conv_2/conv1d_3/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_3�
trunk_/conv_2/conv1d_3/SqueezeSqueezetrunk_/conv_2/conv1d_3:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_3/Squeeze�
&trunk_/conv_2/BiasAdd_3/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_3/ReadVariableOp�
trunk_/conv_2/BiasAdd_3BiasAdd'trunk_/conv_2/conv1d_3/Squeeze:output:0.trunk_/conv_2/BiasAdd_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_3�
trunk_/conv_2/Relu_3Relu trunk_/conv_2/BiasAdd_3:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_3�
trunk_/drop_2/Identity_3Identity"trunk_/conv_2/Relu_3:activations:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/Identity_3�
%trunk_/conv_3/conv1d_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_3/ExpandDims/dim�
!trunk_/conv_3/conv1d_3/ExpandDims
ExpandDims!trunk_/drop_2/Identity_3:output:0.trunk_/conv_3/conv1d_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_3/ExpandDims�
2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_3/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_3/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_3/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_3/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_3/ExpandDims_1�
trunk_/conv_3/conv1d_3Conv2D*trunk_/conv_3/conv1d_3/ExpandDims:output:0,trunk_/conv_3/conv1d_3/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_3�
trunk_/conv_3/conv1d_3/SqueezeSqueezetrunk_/conv_3/conv1d_3:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_3/Squeeze�
&trunk_/conv_3/BiasAdd_3/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_3/ReadVariableOp�
trunk_/conv_3/BiasAdd_3BiasAdd'trunk_/conv_3/conv1d_3/Squeeze:output:0.trunk_/conv_3/BiasAdd_3/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_3�
trunk_/conv_3/Relu_3Relu trunk_/conv_3/BiasAdd_3:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_3�
trunk_/drop_3/Identity_3Identity"trunk_/conv_3/Relu_3:activations:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/Identity_3�
%trunk_/conv_1/conv1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_4/ExpandDims/dim�
!trunk_/conv_1/conv1d_4/ExpandDims
ExpandDimsinputs_2.trunk_/conv_1/conv1d_4/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_4/ExpandDims�
2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_4/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_4/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_4/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_4/ExpandDims_1�
trunk_/conv_1/conv1d_4Conv2D*trunk_/conv_1/conv1d_4/ExpandDims:output:0,trunk_/conv_1/conv1d_4/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_4�
trunk_/conv_1/conv1d_4/SqueezeSqueezetrunk_/conv_1/conv1d_4:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_4/Squeeze�
&trunk_/conv_1/BiasAdd_4/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_4/ReadVariableOp�
trunk_/conv_1/BiasAdd_4BiasAdd'trunk_/conv_1/conv1d_4/Squeeze:output:0.trunk_/conv_1/BiasAdd_4/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_4�
trunk_/conv_1/Relu_4Relu trunk_/conv_1/BiasAdd_4:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_4�
trunk_/drop_1/Identity_4Identity"trunk_/conv_1/Relu_4:activations:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/Identity_4�
%trunk_/conv_2/conv1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_4/ExpandDims/dim�
!trunk_/conv_2/conv1d_4/ExpandDims
ExpandDims!trunk_/drop_1/Identity_4:output:0.trunk_/conv_2/conv1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_4/ExpandDims�
2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_4/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_4/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_4/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_4/ExpandDims_1�
trunk_/conv_2/conv1d_4Conv2D*trunk_/conv_2/conv1d_4/ExpandDims:output:0,trunk_/conv_2/conv1d_4/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_4�
trunk_/conv_2/conv1d_4/SqueezeSqueezetrunk_/conv_2/conv1d_4:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_4/Squeeze�
&trunk_/conv_2/BiasAdd_4/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_4/ReadVariableOp�
trunk_/conv_2/BiasAdd_4BiasAdd'trunk_/conv_2/conv1d_4/Squeeze:output:0.trunk_/conv_2/BiasAdd_4/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_4�
trunk_/conv_2/Relu_4Relu trunk_/conv_2/BiasAdd_4:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_4�
trunk_/drop_2/Identity_4Identity"trunk_/conv_2/Relu_4:activations:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/Identity_4�
%trunk_/conv_3/conv1d_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_4/ExpandDims/dim�
!trunk_/conv_3/conv1d_4/ExpandDims
ExpandDims!trunk_/drop_2/Identity_4:output:0.trunk_/conv_3/conv1d_4/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_4/ExpandDims�
2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_4/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_4/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_4/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_4/ExpandDims_1�
trunk_/conv_3/conv1d_4Conv2D*trunk_/conv_3/conv1d_4/ExpandDims:output:0,trunk_/conv_3/conv1d_4/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_4�
trunk_/conv_3/conv1d_4/SqueezeSqueezetrunk_/conv_3/conv1d_4:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_4/Squeeze�
&trunk_/conv_3/BiasAdd_4/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_4/ReadVariableOp�
trunk_/conv_3/BiasAdd_4BiasAdd'trunk_/conv_3/conv1d_4/Squeeze:output:0.trunk_/conv_3/BiasAdd_4/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_4�
trunk_/conv_3/Relu_4Relu trunk_/conv_3/BiasAdd_4:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_4�
trunk_/drop_3/Identity_4Identity"trunk_/conv_3/Relu_4:activations:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/Identity_4�
%trunk_/conv_1/conv1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_5/ExpandDims/dim�
!trunk_/conv_1/conv1d_5/ExpandDims
ExpandDimsinputs_1.trunk_/conv_1/conv1d_5/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_5/ExpandDims�
2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_5/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_5/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_5/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_5/ExpandDims_1�
trunk_/conv_1/conv1d_5Conv2D*trunk_/conv_1/conv1d_5/ExpandDims:output:0,trunk_/conv_1/conv1d_5/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_5�
trunk_/conv_1/conv1d_5/SqueezeSqueezetrunk_/conv_1/conv1d_5:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_5/Squeeze�
&trunk_/conv_1/BiasAdd_5/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_5/ReadVariableOp�
trunk_/conv_1/BiasAdd_5BiasAdd'trunk_/conv_1/conv1d_5/Squeeze:output:0.trunk_/conv_1/BiasAdd_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_5�
trunk_/conv_1/Relu_5Relu trunk_/conv_1/BiasAdd_5:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_5�
trunk_/drop_1/Identity_5Identity"trunk_/conv_1/Relu_5:activations:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/Identity_5�
%trunk_/conv_2/conv1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_5/ExpandDims/dim�
!trunk_/conv_2/conv1d_5/ExpandDims
ExpandDims!trunk_/drop_1/Identity_5:output:0.trunk_/conv_2/conv1d_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_5/ExpandDims�
2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_5/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_5/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_5/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_5/ExpandDims_1�
trunk_/conv_2/conv1d_5Conv2D*trunk_/conv_2/conv1d_5/ExpandDims:output:0,trunk_/conv_2/conv1d_5/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_5�
trunk_/conv_2/conv1d_5/SqueezeSqueezetrunk_/conv_2/conv1d_5:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_5/Squeeze�
&trunk_/conv_2/BiasAdd_5/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_5/ReadVariableOp�
trunk_/conv_2/BiasAdd_5BiasAdd'trunk_/conv_2/conv1d_5/Squeeze:output:0.trunk_/conv_2/BiasAdd_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_5�
trunk_/conv_2/Relu_5Relu trunk_/conv_2/BiasAdd_5:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_5�
trunk_/drop_2/Identity_5Identity"trunk_/conv_2/Relu_5:activations:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/Identity_5�
%trunk_/conv_3/conv1d_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_5/ExpandDims/dim�
!trunk_/conv_3/conv1d_5/ExpandDims
ExpandDims!trunk_/drop_2/Identity_5:output:0.trunk_/conv_3/conv1d_5/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_5/ExpandDims�
2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_5/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_5/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_5/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_5/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_5/ExpandDims_1�
trunk_/conv_3/conv1d_5Conv2D*trunk_/conv_3/conv1d_5/ExpandDims:output:0,trunk_/conv_3/conv1d_5/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_5�
trunk_/conv_3/conv1d_5/SqueezeSqueezetrunk_/conv_3/conv1d_5:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_5/Squeeze�
&trunk_/conv_3/BiasAdd_5/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_5/ReadVariableOp�
trunk_/conv_3/BiasAdd_5BiasAdd'trunk_/conv_3/conv1d_5/Squeeze:output:0.trunk_/conv_3/BiasAdd_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_5�
trunk_/conv_3/Relu_5Relu trunk_/conv_3/BiasAdd_5:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_5�
trunk_/drop_3/Identity_5Identity"trunk_/conv_3/Relu_5:activations:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/Identity_5�
%trunk_/conv_1/conv1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_1/conv1d_6/ExpandDims/dim�
!trunk_/conv_1/conv1d_6/ExpandDims
ExpandDimsinputs_0.trunk_/conv_1/conv1d_6/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������2#
!trunk_/conv_1/conv1d_6/ExpandDims�
2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype024
2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp�
'trunk_/conv_1/conv1d_6/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_1/conv1d_6/ExpandDims_1/dim�
#trunk_/conv_1/conv1d_6/ExpandDims_1
ExpandDims:trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_1/conv1d_6/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2%
#trunk_/conv_1/conv1d_6/ExpandDims_1�
trunk_/conv_1/conv1d_6Conv2D*trunk_/conv_1/conv1d_6/ExpandDims:output:0,trunk_/conv_1/conv1d_6/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������i *
paddingVALID*
strides
2
trunk_/conv_1/conv1d_6�
trunk_/conv_1/conv1d_6/SqueezeSqueezetrunk_/conv_1/conv1d_6:output:0*
T0*+
_output_shapes
:���������i *
squeeze_dims

���������2 
trunk_/conv_1/conv1d_6/Squeeze�
&trunk_/conv_1/BiasAdd_6/ReadVariableOpReadVariableOp-trunk__conv_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&trunk_/conv_1/BiasAdd_6/ReadVariableOp�
trunk_/conv_1/BiasAdd_6BiasAdd'trunk_/conv_1/conv1d_6/Squeeze:output:0.trunk_/conv_1/BiasAdd_6/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/BiasAdd_6�
trunk_/conv_1/Relu_6Relu trunk_/conv_1/BiasAdd_6:output:0*
T0*+
_output_shapes
:���������i 2
trunk_/conv_1/Relu_6�
trunk_/drop_1/Identity_6Identity"trunk_/conv_1/Relu_6:activations:0*
T0*+
_output_shapes
:���������i 2
trunk_/drop_1/Identity_6�
%trunk_/conv_2/conv1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_2/conv1d_6/ExpandDims/dim�
!trunk_/conv_2/conv1d_6/ExpandDims
ExpandDims!trunk_/drop_1/Identity_6:output:0.trunk_/conv_2/conv1d_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������i 2#
!trunk_/conv_2/conv1d_6/ExpandDims�
2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype024
2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp�
'trunk_/conv_2/conv1d_6/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_2/conv1d_6/ExpandDims_1/dim�
#trunk_/conv_2/conv1d_6/ExpandDims_1
ExpandDims:trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_2/conv1d_6/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2%
#trunk_/conv_2/conv1d_6/ExpandDims_1�
trunk_/conv_2/conv1d_6Conv2D*trunk_/conv_2/conv1d_6/ExpandDims:output:0,trunk_/conv_2/conv1d_6/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������Z@*
paddingVALID*
strides
2
trunk_/conv_2/conv1d_6�
trunk_/conv_2/conv1d_6/SqueezeSqueezetrunk_/conv_2/conv1d_6:output:0*
T0*+
_output_shapes
:���������Z@*
squeeze_dims

���������2 
trunk_/conv_2/conv1d_6/Squeeze�
&trunk_/conv_2/BiasAdd_6/ReadVariableOpReadVariableOp-trunk__conv_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&trunk_/conv_2/BiasAdd_6/ReadVariableOp�
trunk_/conv_2/BiasAdd_6BiasAdd'trunk_/conv_2/conv1d_6/Squeeze:output:0.trunk_/conv_2/BiasAdd_6/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/BiasAdd_6�
trunk_/conv_2/Relu_6Relu trunk_/conv_2/BiasAdd_6:output:0*
T0*+
_output_shapes
:���������Z@2
trunk_/conv_2/Relu_6�
trunk_/drop_2/Identity_6Identity"trunk_/conv_2/Relu_6:activations:0*
T0*+
_output_shapes
:���������Z@2
trunk_/drop_2/Identity_6�
%trunk_/conv_3/conv1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%trunk_/conv_3/conv1d_6/ExpandDims/dim�
!trunk_/conv_3/conv1d_6/ExpandDims
ExpandDims!trunk_/drop_2/Identity_6:output:0.trunk_/conv_3/conv1d_6/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Z@2#
!trunk_/conv_3/conv1d_6/ExpandDims�
2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype024
2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp�
'trunk_/conv_3/conv1d_6/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'trunk_/conv_3/conv1d_6/ExpandDims_1/dim�
#trunk_/conv_3/conv1d_6/ExpandDims_1
ExpandDims:trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp:value:00trunk_/conv_3/conv1d_6/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@`2%
#trunk_/conv_3/conv1d_6/ExpandDims_1�
trunk_/conv_3/conv1d_6Conv2D*trunk_/conv_3/conv1d_6/ExpandDims:output:0,trunk_/conv_3/conv1d_6/ExpandDims_1:output:0*
T0*/
_output_shapes
:���������S`*
paddingVALID*
strides
2
trunk_/conv_3/conv1d_6�
trunk_/conv_3/conv1d_6/SqueezeSqueezetrunk_/conv_3/conv1d_6:output:0*
T0*+
_output_shapes
:���������S`*
squeeze_dims

���������2 
trunk_/conv_3/conv1d_6/Squeeze�
&trunk_/conv_3/BiasAdd_6/ReadVariableOpReadVariableOp-trunk__conv_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02(
&trunk_/conv_3/BiasAdd_6/ReadVariableOp�
trunk_/conv_3/BiasAdd_6BiasAdd'trunk_/conv_3/conv1d_6/Squeeze:output:0.trunk_/conv_3/BiasAdd_6/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/BiasAdd_6�
trunk_/conv_3/Relu_6Relu trunk_/conv_3/BiasAdd_6:output:0*
T0*+
_output_shapes
:���������S`2
trunk_/conv_3/Relu_6�
trunk_/drop_3/Identity_6Identity"trunk_/conv_3/Relu_6:activations:0*
T0*+
_output_shapes
:���������S`2
trunk_/drop_3/Identity_6�
%global_max_pool_/pool_/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%global_max_pool_/pool_/ExpandDims/dim�
!global_max_pool_/pool_/ExpandDims
ExpandDimstrunk_/drop_3/Identity:output:0.global_max_pool_/pool_/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������S`2#
!global_max_pool_/pool_/ExpandDims�
global_max_pool_/pool_/MaxPoolMaxPool*global_max_pool_/pool_/ExpandDims:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2 
global_max_pool_/pool_/MaxPool�
global_max_pool_/pool_/SqueezeSqueeze'global_max_pool_/pool_/MaxPool:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2 
global_max_pool_/pool_/Squeeze�
global_max_pool_/flat_/ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
global_max_pool_/flat_/Const�
global_max_pool_/flat_/ReshapeReshape'global_max_pool_/pool_/Squeeze:output:0%global_max_pool_/flat_/Const:output:0*
T0*(
_output_shapes
:����������2 
global_max_pool_/flat_/Reshape�
'global_max_pool_/pool_/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_1/dim�
#global_max_pool_/pool_/ExpandDims_1
ExpandDims!trunk_/drop_3/Identity_1:output:00global_max_pool_/pool_/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_1�
 global_max_pool_/pool_/MaxPool_1MaxPool,global_max_pool_/pool_/ExpandDims_1:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_1�
 global_max_pool_/pool_/Squeeze_1Squeeze)global_max_pool_/pool_/MaxPool_1:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_1�
global_max_pool_/flat_/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_1�
 global_max_pool_/flat_/Reshape_1Reshape)global_max_pool_/pool_/Squeeze_1:output:0'global_max_pool_/flat_/Const_1:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_1�
'global_max_pool_/pool_/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_2/dim�
#global_max_pool_/pool_/ExpandDims_2
ExpandDims!trunk_/drop_3/Identity_2:output:00global_max_pool_/pool_/ExpandDims_2/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_2�
 global_max_pool_/pool_/MaxPool_2MaxPool,global_max_pool_/pool_/ExpandDims_2:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_2�
 global_max_pool_/pool_/Squeeze_2Squeeze)global_max_pool_/pool_/MaxPool_2:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_2�
global_max_pool_/flat_/Const_2Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_2�
 global_max_pool_/flat_/Reshape_2Reshape)global_max_pool_/pool_/Squeeze_2:output:0'global_max_pool_/flat_/Const_2:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_2�
'global_max_pool_/pool_/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_3/dim�
#global_max_pool_/pool_/ExpandDims_3
ExpandDims!trunk_/drop_3/Identity_3:output:00global_max_pool_/pool_/ExpandDims_3/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_3�
 global_max_pool_/pool_/MaxPool_3MaxPool,global_max_pool_/pool_/ExpandDims_3:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_3�
 global_max_pool_/pool_/Squeeze_3Squeeze)global_max_pool_/pool_/MaxPool_3:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_3�
global_max_pool_/flat_/Const_3Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_3�
 global_max_pool_/flat_/Reshape_3Reshape)global_max_pool_/pool_/Squeeze_3:output:0'global_max_pool_/flat_/Const_3:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_3�
'global_max_pool_/pool_/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_4/dim�
#global_max_pool_/pool_/ExpandDims_4
ExpandDims!trunk_/drop_3/Identity_4:output:00global_max_pool_/pool_/ExpandDims_4/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_4�
 global_max_pool_/pool_/MaxPool_4MaxPool,global_max_pool_/pool_/ExpandDims_4:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_4�
 global_max_pool_/pool_/Squeeze_4Squeeze)global_max_pool_/pool_/MaxPool_4:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_4�
global_max_pool_/flat_/Const_4Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_4�
 global_max_pool_/flat_/Reshape_4Reshape)global_max_pool_/pool_/Squeeze_4:output:0'global_max_pool_/flat_/Const_4:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_4�
'global_max_pool_/pool_/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_5/dim�
#global_max_pool_/pool_/ExpandDims_5
ExpandDims!trunk_/drop_3/Identity_5:output:00global_max_pool_/pool_/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_5�
 global_max_pool_/pool_/MaxPool_5MaxPool,global_max_pool_/pool_/ExpandDims_5:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_5�
 global_max_pool_/pool_/Squeeze_5Squeeze)global_max_pool_/pool_/MaxPool_5:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_5�
global_max_pool_/flat_/Const_5Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_5�
 global_max_pool_/flat_/Reshape_5Reshape)global_max_pool_/pool_/Squeeze_5:output:0'global_max_pool_/flat_/Const_5:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_5�
'global_max_pool_/pool_/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'global_max_pool_/pool_/ExpandDims_6/dim�
#global_max_pool_/pool_/ExpandDims_6
ExpandDims!trunk_/drop_3/Identity_6:output:00global_max_pool_/pool_/ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:���������S`2%
#global_max_pool_/pool_/ExpandDims_6�
 global_max_pool_/pool_/MaxPool_6MaxPool,global_max_pool_/pool_/ExpandDims_6:output:0*/
_output_shapes
:���������)`*
ksize
*
paddingVALID*
strides
2"
 global_max_pool_/pool_/MaxPool_6�
 global_max_pool_/pool_/Squeeze_6Squeeze)global_max_pool_/pool_/MaxPool_6:output:0*
T0*+
_output_shapes
:���������)`*
squeeze_dims
2"
 global_max_pool_/pool_/Squeeze_6�
global_max_pool_/flat_/Const_6Const*
_output_shapes
:*
dtype0*
valueB"����`  2 
global_max_pool_/flat_/Const_6�
 global_max_pool_/flat_/Reshape_6Reshape)global_max_pool_/pool_/Squeeze_6:output:0'global_max_pool_/flat_/Const_6:output:0*
T0*(
_output_shapes
:����������2"
 global_max_pool_/flat_/Reshape_6�
dens_7/MatMul/ReadVariableOpReadVariableOp%dens_7_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_7/MatMul/ReadVariableOp�
dens_7/MatMulMatMul'global_max_pool_/flat_/Reshape:output:0$dens_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_7/MatMul�
dens_7/BiasAdd/ReadVariableOpReadVariableOp&dens_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_7/BiasAdd/ReadVariableOp�
dens_7/BiasAddBiasAdddens_7/MatMul:product:0%dens_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_7/BiasAddn
dens_7/ReluReludens_7/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_7/Relu�
dens_6/MatMul/ReadVariableOpReadVariableOp%dens_6_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_6/MatMul/ReadVariableOp�
dens_6/MatMulMatMul)global_max_pool_/flat_/Reshape_1:output:0$dens_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_6/MatMul�
dens_6/BiasAdd/ReadVariableOpReadVariableOp&dens_6_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_6/BiasAdd/ReadVariableOp�
dens_6/BiasAddBiasAdddens_6/MatMul:product:0%dens_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_6/BiasAddn
dens_6/ReluReludens_6/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_6/Relu�
dens_5/MatMul/ReadVariableOpReadVariableOp%dens_5_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_5/MatMul/ReadVariableOp�
dens_5/MatMulMatMul)global_max_pool_/flat_/Reshape_2:output:0$dens_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_5/MatMul�
dens_5/BiasAdd/ReadVariableOpReadVariableOp&dens_5_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_5/BiasAdd/ReadVariableOp�
dens_5/BiasAddBiasAdddens_5/MatMul:product:0%dens_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_5/BiasAddn
dens_5/ReluReludens_5/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_5/Relu�
dens_4/MatMul/ReadVariableOpReadVariableOp%dens_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_4/MatMul/ReadVariableOp�
dens_4/MatMulMatMul)global_max_pool_/flat_/Reshape_3:output:0$dens_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_4/MatMul�
dens_4/BiasAdd/ReadVariableOpReadVariableOp&dens_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_4/BiasAdd/ReadVariableOp�
dens_4/BiasAddBiasAdddens_4/MatMul:product:0%dens_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_4/BiasAddn
dens_4/ReluReludens_4/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_4/Relu�
dens_3/MatMul/ReadVariableOpReadVariableOp%dens_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_3/MatMul/ReadVariableOp�
dens_3/MatMulMatMul)global_max_pool_/flat_/Reshape_4:output:0$dens_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_3/MatMul�
dens_3/BiasAdd/ReadVariableOpReadVariableOp&dens_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_3/BiasAdd/ReadVariableOp�
dens_3/BiasAddBiasAdddens_3/MatMul:product:0%dens_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_3/BiasAddn
dens_3/ReluReludens_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_3/Relu�
dens_2/MatMul/ReadVariableOpReadVariableOp%dens_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_2/MatMul/ReadVariableOp�
dens_2/MatMulMatMul)global_max_pool_/flat_/Reshape_5:output:0$dens_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_2/MatMul�
dens_2/BiasAdd/ReadVariableOpReadVariableOp&dens_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_2/BiasAdd/ReadVariableOp�
dens_2/BiasAddBiasAdddens_2/MatMul:product:0%dens_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_2/BiasAddn
dens_2/ReluReludens_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_2/Relu�
dens_1/MatMul/ReadVariableOpReadVariableOp%dens_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dens_1/MatMul/ReadVariableOp�
dens_1/MatMulMatMul)global_max_pool_/flat_/Reshape_6:output:0$dens_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_1/MatMul�
dens_1/BiasAdd/ReadVariableOpReadVariableOp&dens_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dens_1/BiasAdd/ReadVariableOp�
dens_1/BiasAddBiasAdddens_1/MatMul:product:0%dens_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dens_1/BiasAddn
dens_1/ReluReludens_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dens_1/Relu�
head_7/MatMul/ReadVariableOpReadVariableOp%head_7_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_7/MatMul/ReadVariableOp�
head_7/MatMulMatMuldens_7/Relu:activations:0$head_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_7/MatMul�
head_7/BiasAdd/ReadVariableOpReadVariableOp&head_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_7/BiasAdd/ReadVariableOp�
head_7/BiasAddBiasAddhead_7/MatMul:product:0%head_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_7/BiasAddv
head_7/SigmoidSigmoidhead_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_7/Sigmoid�
head_6/MatMul/ReadVariableOpReadVariableOp%head_6_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_6/MatMul/ReadVariableOp�
head_6/MatMulMatMuldens_6/Relu:activations:0$head_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_6/MatMul�
head_6/BiasAdd/ReadVariableOpReadVariableOp&head_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_6/BiasAdd/ReadVariableOp�
head_6/BiasAddBiasAddhead_6/MatMul:product:0%head_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_6/BiasAddv
head_6/SigmoidSigmoidhead_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_6/Sigmoid�
head_5/MatMul/ReadVariableOpReadVariableOp%head_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_5/MatMul/ReadVariableOp�
head_5/MatMulMatMuldens_5/Relu:activations:0$head_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_5/MatMul�
head_5/BiasAdd/ReadVariableOpReadVariableOp&head_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_5/BiasAdd/ReadVariableOp�
head_5/BiasAddBiasAddhead_5/MatMul:product:0%head_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_5/BiasAddv
head_5/SigmoidSigmoidhead_5/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_5/Sigmoid�
head_4/MatMul/ReadVariableOpReadVariableOp%head_4_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_4/MatMul/ReadVariableOp�
head_4/MatMulMatMuldens_4/Relu:activations:0$head_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_4/MatMul�
head_4/BiasAdd/ReadVariableOpReadVariableOp&head_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_4/BiasAdd/ReadVariableOp�
head_4/BiasAddBiasAddhead_4/MatMul:product:0%head_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_4/BiasAddv
head_4/SigmoidSigmoidhead_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_4/Sigmoid�
head_3/MatMul/ReadVariableOpReadVariableOp%head_3_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_3/MatMul/ReadVariableOp�
head_3/MatMulMatMuldens_3/Relu:activations:0$head_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_3/MatMul�
head_3/BiasAdd/ReadVariableOpReadVariableOp&head_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_3/BiasAdd/ReadVariableOp�
head_3/BiasAddBiasAddhead_3/MatMul:product:0%head_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_3/BiasAddv
head_3/SigmoidSigmoidhead_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_3/Sigmoid�
head_2/MatMul/ReadVariableOpReadVariableOp%head_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_2/MatMul/ReadVariableOp�
head_2/MatMulMatMuldens_2/Relu:activations:0$head_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_2/MatMul�
head_2/BiasAdd/ReadVariableOpReadVariableOp&head_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_2/BiasAdd/ReadVariableOp�
head_2/BiasAddBiasAddhead_2/MatMul:product:0%head_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_2/BiasAddv
head_2/SigmoidSigmoidhead_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_2/Sigmoid�
head_1/MatMul/ReadVariableOpReadVariableOp%head_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
head_1/MatMul/ReadVariableOp�
head_1/MatMulMatMuldens_1/Relu:activations:0$head_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_1/MatMul�
head_1/BiasAdd/ReadVariableOpReadVariableOp&head_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
head_1/BiasAdd/ReadVariableOp�
head_1/BiasAddBiasAddhead_1/MatMul:product:0%head_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
head_1/BiasAddv
head_1/SigmoidSigmoidhead_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
head_1/Sigmoid�
/conv_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9trunk__conv_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv_1/kernel/Regularizer/Square/ReadVariableOp�
 conv_1/kernel/Regularizer/SquareSquare7conv_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv_1/kernel/Regularizer/Square�
conv_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_1/kernel/Regularizer/Const�
conv_1/kernel/Regularizer/SumSum$conv_1/kernel/Regularizer/Square:y:0(conv_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/Sum�
conv_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_1/kernel/Regularizer/mul/x�
conv_1/kernel/Regularizer/mulMul(conv_1/kernel/Regularizer/mul/x:output:0&conv_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_1/kernel/Regularizer/mul�
/conv_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9trunk__conv_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype021
/conv_2/kernel/Regularizer/Square/ReadVariableOp�
 conv_2/kernel/Regularizer/SquareSquare7conv_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: @2"
 conv_2/kernel/Regularizer/Square�
conv_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_2/kernel/Regularizer/Const�
conv_2/kernel/Regularizer/SumSum$conv_2/kernel/Regularizer/Square:y:0(conv_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/Sum�
conv_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_2/kernel/Regularizer/mul/x�
conv_2/kernel/Regularizer/mulMul(conv_2/kernel/Regularizer/mul/x:output:0&conv_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_2/kernel/Regularizer/mul�
/conv_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9trunk__conv_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@`*
dtype021
/conv_3/kernel/Regularizer/Square/ReadVariableOp�
 conv_3/kernel/Regularizer/SquareSquare7conv_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
:@`2"
 conv_3/kernel/Regularizer/Square�
conv_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv_3/kernel/Regularizer/Const�
conv_3/kernel/Regularizer/SumSum$conv_3/kernel/Regularizer/Square:y:0(conv_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/Sum�
conv_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��82!
conv_3/kernel/Regularizer/mul/x�
conv_3/kernel/Regularizer/mulMul(conv_3/kernel/Regularizer/mul/x:output:0&conv_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv_3/kernel/Regularizer/mul�
IdentityIdentityhead_1/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity�

Identity_1Identityhead_2/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_1�

Identity_2Identityhead_3/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_2�

Identity_3Identityhead_4/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_3�

Identity_4Identityhead_5/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_4�

Identity_5Identityhead_6/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_5�

Identity_6Identityhead_7/Sigmoid:y:00^conv_1/kernel/Regularizer/Square/ReadVariableOp0^conv_2/kernel/Regularizer/Square/ReadVariableOp0^conv_3/kernel/Regularizer/Square/ReadVariableOp^dens_1/BiasAdd/ReadVariableOp^dens_1/MatMul/ReadVariableOp^dens_2/BiasAdd/ReadVariableOp^dens_2/MatMul/ReadVariableOp^dens_3/BiasAdd/ReadVariableOp^dens_3/MatMul/ReadVariableOp^dens_4/BiasAdd/ReadVariableOp^dens_4/MatMul/ReadVariableOp^dens_5/BiasAdd/ReadVariableOp^dens_5/MatMul/ReadVariableOp^dens_6/BiasAdd/ReadVariableOp^dens_6/MatMul/ReadVariableOp^dens_7/BiasAdd/ReadVariableOp^dens_7/MatMul/ReadVariableOp^head_1/BiasAdd/ReadVariableOp^head_1/MatMul/ReadVariableOp^head_2/BiasAdd/ReadVariableOp^head_2/MatMul/ReadVariableOp^head_3/BiasAdd/ReadVariableOp^head_3/MatMul/ReadVariableOp^head_4/BiasAdd/ReadVariableOp^head_4/MatMul/ReadVariableOp^head_5/BiasAdd/ReadVariableOp^head_5/MatMul/ReadVariableOp^head_6/BiasAdd/ReadVariableOp^head_6/MatMul/ReadVariableOp^head_7/BiasAdd/ReadVariableOp^head_7/MatMul/ReadVariableOp%^trunk_/conv_1/BiasAdd/ReadVariableOp'^trunk_/conv_1/BiasAdd_1/ReadVariableOp'^trunk_/conv_1/BiasAdd_2/ReadVariableOp'^trunk_/conv_1/BiasAdd_3/ReadVariableOp'^trunk_/conv_1/BiasAdd_4/ReadVariableOp'^trunk_/conv_1/BiasAdd_5/ReadVariableOp'^trunk_/conv_1/BiasAdd_6/ReadVariableOp1^trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_2/BiasAdd/ReadVariableOp'^trunk_/conv_2/BiasAdd_1/ReadVariableOp'^trunk_/conv_2/BiasAdd_2/ReadVariableOp'^trunk_/conv_2/BiasAdd_3/ReadVariableOp'^trunk_/conv_2/BiasAdd_4/ReadVariableOp'^trunk_/conv_2/BiasAdd_5/ReadVariableOp'^trunk_/conv_2/BiasAdd_6/ReadVariableOp1^trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp%^trunk_/conv_3/BiasAdd/ReadVariableOp'^trunk_/conv_3/BiasAdd_1/ReadVariableOp'^trunk_/conv_3/BiasAdd_2/ReadVariableOp'^trunk_/conv_3/BiasAdd_3/ReadVariableOp'^trunk_/conv_3/BiasAdd_4/ReadVariableOp'^trunk_/conv_3/BiasAdd_5/ReadVariableOp'^trunk_/conv_3/BiasAdd_6/ReadVariableOp1^trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp3^trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity_6"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������:����������:����������:����������:����������:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/conv_1/kernel/Regularizer/Square/ReadVariableOp/conv_1/kernel/Regularizer/Square/ReadVariableOp2b
/conv_2/kernel/Regularizer/Square/ReadVariableOp/conv_2/kernel/Regularizer/Square/ReadVariableOp2b
/conv_3/kernel/Regularizer/Square/ReadVariableOp/conv_3/kernel/Regularizer/Square/ReadVariableOp2>
dens_1/BiasAdd/ReadVariableOpdens_1/BiasAdd/ReadVariableOp2<
dens_1/MatMul/ReadVariableOpdens_1/MatMul/ReadVariableOp2>
dens_2/BiasAdd/ReadVariableOpdens_2/BiasAdd/ReadVariableOp2<
dens_2/MatMul/ReadVariableOpdens_2/MatMul/ReadVariableOp2>
dens_3/BiasAdd/ReadVariableOpdens_3/BiasAdd/ReadVariableOp2<
dens_3/MatMul/ReadVariableOpdens_3/MatMul/ReadVariableOp2>
dens_4/BiasAdd/ReadVariableOpdens_4/BiasAdd/ReadVariableOp2<
dens_4/MatMul/ReadVariableOpdens_4/MatMul/ReadVariableOp2>
dens_5/BiasAdd/ReadVariableOpdens_5/BiasAdd/ReadVariableOp2<
dens_5/MatMul/ReadVariableOpdens_5/MatMul/ReadVariableOp2>
dens_6/BiasAdd/ReadVariableOpdens_6/BiasAdd/ReadVariableOp2<
dens_6/MatMul/ReadVariableOpdens_6/MatMul/ReadVariableOp2>
dens_7/BiasAdd/ReadVariableOpdens_7/BiasAdd/ReadVariableOp2<
dens_7/MatMul/ReadVariableOpdens_7/MatMul/ReadVariableOp2>
head_1/BiasAdd/ReadVariableOphead_1/BiasAdd/ReadVariableOp2<
head_1/MatMul/ReadVariableOphead_1/MatMul/ReadVariableOp2>
head_2/BiasAdd/ReadVariableOphead_2/BiasAdd/ReadVariableOp2<
head_2/MatMul/ReadVariableOphead_2/MatMul/ReadVariableOp2>
head_3/BiasAdd/ReadVariableOphead_3/BiasAdd/ReadVariableOp2<
head_3/MatMul/ReadVariableOphead_3/MatMul/ReadVariableOp2>
head_4/BiasAdd/ReadVariableOphead_4/BiasAdd/ReadVariableOp2<
head_4/MatMul/ReadVariableOphead_4/MatMul/ReadVariableOp2>
head_5/BiasAdd/ReadVariableOphead_5/BiasAdd/ReadVariableOp2<
head_5/MatMul/ReadVariableOphead_5/MatMul/ReadVariableOp2>
head_6/BiasAdd/ReadVariableOphead_6/BiasAdd/ReadVariableOp2<
head_6/MatMul/ReadVariableOphead_6/MatMul/ReadVariableOp2>
head_7/BiasAdd/ReadVariableOphead_7/BiasAdd/ReadVariableOp2<
head_7/MatMul/ReadVariableOphead_7/MatMul/ReadVariableOp2L
$trunk_/conv_1/BiasAdd/ReadVariableOp$trunk_/conv_1/BiasAdd/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_1/ReadVariableOp&trunk_/conv_1/BiasAdd_1/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_2/ReadVariableOp&trunk_/conv_1/BiasAdd_2/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_3/ReadVariableOp&trunk_/conv_1/BiasAdd_3/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_4/ReadVariableOp&trunk_/conv_1/BiasAdd_4/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_5/ReadVariableOp&trunk_/conv_1/BiasAdd_5/ReadVariableOp2P
&trunk_/conv_1/BiasAdd_6/ReadVariableOp&trunk_/conv_1/BiasAdd_6/ReadVariableOp2d
0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp0trunk_/conv_1/conv1d/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_1/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_2/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_3/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_4/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_5/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp2trunk_/conv_1/conv1d_6/ExpandDims_1/ReadVariableOp2L
$trunk_/conv_2/BiasAdd/ReadVariableOp$trunk_/conv_2/BiasAdd/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_1/ReadVariableOp&trunk_/conv_2/BiasAdd_1/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_2/ReadVariableOp&trunk_/conv_2/BiasAdd_2/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_3/ReadVariableOp&trunk_/conv_2/BiasAdd_3/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_4/ReadVariableOp&trunk_/conv_2/BiasAdd_4/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_5/ReadVariableOp&trunk_/conv_2/BiasAdd_5/ReadVariableOp2P
&trunk_/conv_2/BiasAdd_6/ReadVariableOp&trunk_/conv_2/BiasAdd_6/ReadVariableOp2d
0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp0trunk_/conv_2/conv1d/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_1/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_2/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_3/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_4/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_5/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp2trunk_/conv_2/conv1d_6/ExpandDims_1/ReadVariableOp2L
$trunk_/conv_3/BiasAdd/ReadVariableOp$trunk_/conv_3/BiasAdd/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_1/ReadVariableOp&trunk_/conv_3/BiasAdd_1/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_2/ReadVariableOp&trunk_/conv_3/BiasAdd_2/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_3/ReadVariableOp&trunk_/conv_3/BiasAdd_3/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_4/ReadVariableOp&trunk_/conv_3/BiasAdd_4/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_5/ReadVariableOp&trunk_/conv_3/BiasAdd_5/ReadVariableOp2P
&trunk_/conv_3/BiasAdd_6/ReadVariableOp&trunk_/conv_3/BiasAdd_6/ReadVariableOp2d
0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp0trunk_/conv_3/conv1d/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_1/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_2/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_3/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_4/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_5/ExpandDims_1/ReadVariableOp2h
2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp2trunk_/conv_3/conv1d_6/ExpandDims_1/ReadVariableOp:V R
,
_output_shapes
:����������
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/2:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/3:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/4:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/5:VR
,
_output_shapes
:����������
"
_user_specified_name
inputs/6"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
@
input_15
serving_default_input_1:0����������
@
input_25
serving_default_input_2:0����������
@
input_35
serving_default_input_3:0����������
@
input_45
serving_default_input_4:0����������
@
input_55
serving_default_input_5:0����������
@
input_65
serving_default_input_6:0����������
@
input_75
serving_default_input_7:0����������:
head_10
StatefulPartitionedCall:0���������:
head_20
StatefulPartitionedCall:1���������:
head_30
StatefulPartitionedCall:2���������:
head_40
StatefulPartitionedCall:3���������:
head_50
StatefulPartitionedCall:4���������:
head_60
StatefulPartitionedCall:5���������:
head_70
StatefulPartitionedCall:6���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer-8

layer_with_weights-1

layer-9
layer_with_weights-2
layer-10
layer_with_weights-3
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer_with_weights-10
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer_with_weights-14
layer-22
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"ˁ
_tf_keras_network��{"name": "multi-task_self-supervised", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "multi-task_self-supervised", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "trunk_", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_"}, "name": "input_", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["drop_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_2", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["drop_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}], "input_layers": [["input_", 0, 0]], "output_layers": [["drop_3", 0, 0]]}, "name": "trunk_", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]], [["input_3", 0, 0, {}]], [["input_4", 0, 0, {}]], [["input_5", 0, 0, {}]], [["input_6", 0, 0, {}]], [["input_7", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "global_max_pool_", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_I"}, "name": "input_I", "inbound_nodes": []}, {"class_name": "MaxPooling1D", "config": {"name": "pool_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "pool_", "inbound_nodes": [[["input_I", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flat_", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flat_", "inbound_nodes": [[["pool_", 0, 0, {}]]]}], "input_layers": [["input_I", 0, 0]], "output_layers": [["flat_", 0, 0]]}, "name": "global_max_pool_", "inbound_nodes": [[["trunk_", 1, 0, {}]], [["trunk_", 2, 0, {}]], [["trunk_", 3, 0, {}]], [["trunk_", 4, 0, {}]], [["trunk_", 5, 0, {}]], [["trunk_", 6, 0, {}]], [["trunk_", 7, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dens_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_1", "inbound_nodes": [[["global_max_pool_", 1, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dens_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_2", "inbound_nodes": [[["global_max_pool_", 2, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dens_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_3", "inbound_nodes": [[["global_max_pool_", 3, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dens_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_4", "inbound_nodes": [[["global_max_pool_", 4, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dens_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_5", "inbound_nodes": [[["global_max_pool_", 5, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dens_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_6", "inbound_nodes": [[["global_max_pool_", 6, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dens_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_7", "inbound_nodes": [[["global_max_pool_", 7, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "head_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_1", "inbound_nodes": [[["dens_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "head_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_2", "inbound_nodes": [[["dens_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "head_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_3", "inbound_nodes": [[["dens_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "head_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_4", "inbound_nodes": [[["dens_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "head_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_5", "inbound_nodes": [[["dens_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "head_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_6", "inbound_nodes": [[["dens_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "head_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_7", "inbound_nodes": [[["dens_7", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0], ["input_4", 0, 0], ["input_5", 0, 0], ["input_6", 0, 0], ["input_7", 0, 0]], "output_layers": [["head_1", 0, 0], ["head_2", 0, 0], ["head_3", 0, 0], ["head_4", 0, 0], ["head_5", 0, 0], ["head_6", 0, 0], ["head_7", 0, 0]]}, "shared_object_id": 70, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 3]}, {"class_name": "TensorShape", "items": [null, 128, 3]}, {"class_name": "TensorShape", "items": [null, 128, 3]}, {"class_name": "TensorShape", "items": [null, 128, 3]}, {"class_name": "TensorShape", "items": [null, 128, 3]}, {"class_name": "TensorShape", "items": [null, 128, 3]}, {"class_name": "TensorShape", "items": [null, 128, 3]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 3]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 3]}, "float32", "input_2"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 3]}, "float32", "input_3"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 3]}, "float32", "input_4"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 3]}, "float32", "input_5"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 3]}, "float32", "input_6"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 3]}, "float32", "input_7"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "multi-task_self-supervised", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "Functional", "config": {"name": "trunk_", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_"}, "name": "input_", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["drop_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_2", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["drop_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}], "input_layers": [["input_", 0, 0]], "output_layers": [["drop_3", 0, 0]]}, "name": "trunk_", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]], [["input_3", 0, 0, {}]], [["input_4", 0, 0, {}]], [["input_5", 0, 0, {}]], [["input_6", 0, 0, {}]], [["input_7", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Functional", "config": {"name": "global_max_pool_", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_I"}, "name": "input_I", "inbound_nodes": []}, {"class_name": "MaxPooling1D", "config": {"name": "pool_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "pool_", "inbound_nodes": [[["input_I", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flat_", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flat_", "inbound_nodes": [[["pool_", 0, 0, {}]]]}], "input_layers": [["input_I", 0, 0]], "output_layers": [["flat_", 0, 0]]}, "name": "global_max_pool_", "inbound_nodes": [[["trunk_", 1, 0, {}]], [["trunk_", 2, 0, {}]], [["trunk_", 3, 0, {}]], [["trunk_", 4, 0, {}]], [["trunk_", 5, 0, {}]], [["trunk_", 6, 0, {}]], [["trunk_", 7, 0, {}]]], "shared_object_id": 27}, {"class_name": "Dense", "config": {"name": "dens_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_1", "inbound_nodes": [[["global_max_pool_", 1, 0, {}]]], "shared_object_id": 30}, {"class_name": "Dense", "config": {"name": "dens_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_2", "inbound_nodes": [[["global_max_pool_", 2, 0, {}]]], "shared_object_id": 33}, {"class_name": "Dense", "config": {"name": "dens_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_3", "inbound_nodes": [[["global_max_pool_", 3, 0, {}]]], "shared_object_id": 36}, {"class_name": "Dense", "config": {"name": "dens_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_4", "inbound_nodes": [[["global_max_pool_", 4, 0, {}]]], "shared_object_id": 39}, {"class_name": "Dense", "config": {"name": "dens_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_5", "inbound_nodes": [[["global_max_pool_", 5, 0, {}]]], "shared_object_id": 42}, {"class_name": "Dense", "config": {"name": "dens_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 43}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_6", "inbound_nodes": [[["global_max_pool_", 6, 0, {}]]], "shared_object_id": 45}, {"class_name": "Dense", "config": {"name": "dens_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 46}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 47}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dens_7", "inbound_nodes": [[["global_max_pool_", 7, 0, {}]]], "shared_object_id": 48}, {"class_name": "Dense", "config": {"name": "head_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_1", "inbound_nodes": [[["dens_1", 0, 0, {}]]], "shared_object_id": 51}, {"class_name": "Dense", "config": {"name": "head_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_2", "inbound_nodes": [[["dens_2", 0, 0, {}]]], "shared_object_id": 54}, {"class_name": "Dense", "config": {"name": "head_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_3", "inbound_nodes": [[["dens_3", 0, 0, {}]]], "shared_object_id": 57}, {"class_name": "Dense", "config": {"name": "head_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 59}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_4", "inbound_nodes": [[["dens_4", 0, 0, {}]]], "shared_object_id": 60}, {"class_name": "Dense", "config": {"name": "head_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_5", "inbound_nodes": [[["dens_5", 0, 0, {}]]], "shared_object_id": 63}, {"class_name": "Dense", "config": {"name": "head_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 64}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 65}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_6", "inbound_nodes": [[["dens_6", 0, 0, {}]]], "shared_object_id": 66}, {"class_name": "Dense", "config": {"name": "head_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "head_7", "inbound_nodes": [[["dens_7", 0, 0, {}]]], "shared_object_id": 69}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0], ["input_3", 0, 0], ["input_4", 0, 0], ["input_5", 0, 0], ["input_6", 0, 0], ["input_7", 0, 0]], "output_layers": [["head_1", 0, 0], ["head_2", 0, 0], ["head_3", 0, 0], ["head_4", 0, 0], ["head_5", 0, 0], ["head_6", 0, 0], ["head_7", 0, 0]]}}, "training_config": {"loss": ["binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy", "binary_crossentropy"], "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "head_1_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 78}], [{"class_name": "MeanMetricWrapper", "config": {"name": "head_2_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 79}], [{"class_name": "MeanMetricWrapper", "config": {"name": "head_3_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 80}], [{"class_name": "MeanMetricWrapper", "config": {"name": "head_4_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 81}], [{"class_name": "MeanMetricWrapper", "config": {"name": "head_5_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 82}], [{"class_name": "MeanMetricWrapper", "config": {"name": "head_6_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 83}], [{"class_name": "MeanMetricWrapper", "config": {"name": "head_7_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 84}]], "weighted_metrics": null, "loss_weights": [0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285], "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0003000000142492354, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
�@
layer-0
 layer_with_weights-0
 layer-1
!layer-2
"layer_with_weights-1
"layer-3
#layer-4
$layer_with_weights-2
$layer-5
%layer-6
&	variables
'trainable_variables
(regularization_losses
)	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�>
_tf_keras_network�>{"name": "trunk_", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "trunk_", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_"}, "name": "input_", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["drop_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_2", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["drop_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}], "input_layers": [["input_", 0, 0]], "output_layers": [["drop_3", 0, 0]]}, "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]], [["input_3", 0, 0, {}]], [["input_4", 0, 0, {}]], [["input_5", 0, 0, {}]], [["input_6", 0, 0, {}]], [["input_7", 0, 0, {}]]], "shared_object_id": 23, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 3]}, "float32", "input_"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "trunk_", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_"}, "name": "input_", "inbound_nodes": [], "shared_object_id": 7}, {"class_name": "Conv1D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}, "shared_object_id": 10}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "drop_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_1", "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv1D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["drop_1", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Dropout", "config": {"name": "drop_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_2", "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}, "shared_object_id": 20}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["drop_2", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "drop_3", "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 22}], "input_layers": [["input_", 0, 0]], "output_layers": [["drop_3", 0, 0]]}}}
�
*layer-0
+layer-1
,layer-2
-	variables
.trainable_variables
/regularization_losses
0	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_network�{"name": "global_max_pool_", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "global_max_pool_", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_I"}, "name": "input_I", "inbound_nodes": []}, {"class_name": "MaxPooling1D", "config": {"name": "pool_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "pool_", "inbound_nodes": [[["input_I", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flat_", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flat_", "inbound_nodes": [[["pool_", 0, 0, {}]]]}], "input_layers": [["input_I", 0, 0]], "output_layers": [["flat_", 0, 0]]}, "inbound_nodes": [[["trunk_", 1, 0, {}]], [["trunk_", 2, 0, {}]], [["trunk_", 3, 0, {}]], [["trunk_", 4, 0, {}]], [["trunk_", 5, 0, {}]], [["trunk_", 6, 0, {}]], [["trunk_", 7, 0, {}]]], "shared_object_id": 27, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 83, 96]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 83, 96]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 83, 96]}, "float32", "input_I"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "global_max_pool_", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_I"}, "name": "input_I", "inbound_nodes": [], "shared_object_id": 24}, {"class_name": "MaxPooling1D", "config": {"name": "pool_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "pool_", "inbound_nodes": [[["input_I", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "Flatten", "config": {"name": "flat_", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flat_", "inbound_nodes": [[["pool_", 0, 0, {}]]], "shared_object_id": 26}], "input_layers": [["input_I", 0, 0]], "output_layers": [["flat_", 0, 0]]}}}
�	

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dens_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dens_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["global_max_pool_", 1, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3936}}, "shared_object_id": 87}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3936]}}
�	

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dens_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dens_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["global_max_pool_", 2, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3936}}, "shared_object_id": 88}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3936]}}
�	

=kernel
>bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dens_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dens_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["global_max_pool_", 3, 0, {}]]], "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3936}}, "shared_object_id": 89}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3936]}}
�	

Ckernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dens_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dens_4", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["global_max_pool_", 4, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3936}}, "shared_object_id": 90}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3936]}}
�	

Ikernel
Jbias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dens_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dens_5", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["global_max_pool_", 5, 0, {}]]], "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3936}}, "shared_object_id": 91}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3936]}}
�	

Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dens_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dens_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 43}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 44}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["global_max_pool_", 6, 0, {}]]], "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3936}}, "shared_object_id": 92}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3936]}}
�	

Ukernel
Vbias
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dens_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dens_7", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 46}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 47}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["global_max_pool_", 7, 0, {}]]], "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3936}}, "shared_object_id": 93}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3936]}}
�

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "head_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "head_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dens_1", 0, 0, {}]]], "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 94}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "head_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "head_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dens_2", 0, 0, {}]]], "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 95}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "head_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "head_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 55}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 56}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dens_3", 0, 0, {}]]], "shared_object_id": 57, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 96}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "head_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "head_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 58}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 59}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dens_4", 0, 0, {}]]], "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 97}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

skernel
tbias
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "head_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "head_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dens_5", 0, 0, {}]]], "shared_object_id": 63, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 98}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "head_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "head_6", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 64}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 65}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dens_6", 0, 0, {}]]], "shared_object_id": 66, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 99}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�	

kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "head_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "head_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dens_7", 0, 0, {}]]], "shared_object_id": 69, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 100}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate1m�2m�7m�8m�=m�>m�Cm�Dm�Im�Jm�Om�Pm�Um�Vm�[m�\m�am�bm�gm�hm�mm�nm�sm�tm�ym�zm�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�1v�2v�7v�8v�=v�>v�Cv�Dv�Iv�Jv�Ov�Pv�Uv�Vv�[v�\v�av�bv�gv�hv�mv�nv�sv�tv�yv�zv�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
 "
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
16
27
78
89
=10
>11
C12
D13
I14
J15
O16
P17
U18
V19
[20
\21
a22
b23
g24
h25
m26
n27
s28
t29
y30
z31
32
�33"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
16
27
78
89
=10
>11
C12
D13
I14
J15
O16
P17
U18
V19
[20
\21
a22
b23
g24
h25
m26
n27
s28
t29
y30
z31
32
�33"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
	variables
trainable_variables
�layers
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_"}}
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�

_tf_keras_layer�	{"name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}, "shared_object_id": 10}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 3}}, "shared_object_id": 101}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 3]}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "drop_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 12}
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�

_tf_keras_layer�
{"name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [16]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["drop_1", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}, "shared_object_id": 102}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 105, 32]}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "drop_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 17}
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�

_tf_keras_layer�
{"name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}, "shared_object_id": 20}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["drop_2", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}, "shared_object_id": 103}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90, 64]}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "drop_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "drop_3", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 22}
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
&	variables
'trainable_variables
�layers
(regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_I", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 96]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 96]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_I"}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "pool_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "pool_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["input_I", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 104}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "flat_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flat_", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["pool_", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 105}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
-	variables
.trainable_variables
�layers
/regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2dens_1/kernel
:�2dens_1/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
3	variables
4trainable_variables
�layers
5regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2dens_2/kernel
:�2dens_2/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
9	variables
:trainable_variables
�layers
;regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2dens_3/kernel
:�2dens_3/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
?	variables
@trainable_variables
�layers
Aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2dens_4/kernel
:�2dens_4/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
E	variables
Ftrainable_variables
�layers
Gregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2dens_5/kernel
:�2dens_5/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
K	variables
Ltrainable_variables
�layers
Mregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2dens_6/kernel
:�2dens_6/bias
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
Q	variables
Rtrainable_variables
�layers
Sregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:
��2dens_7/kernel
:�2dens_7/bias
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
W	variables
Xtrainable_variables
�layers
Yregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2head_1/kernel
:2head_1/bias
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
]	variables
^trainable_variables
�layers
_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2head_2/kernel
:2head_2/bias
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
c	variables
dtrainable_variables
�layers
eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2head_3/kernel
:2head_3/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
i	variables
jtrainable_variables
�layers
kregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2head_4/kernel
:2head_4/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
o	variables
ptrainable_variables
�layers
qregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2head_5/kernel
:2head_5/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
u	variables
vtrainable_variables
�layers
wregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2head_6/kernel
:2head_6/bias
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
{	variables
|trainable_variables
�layers
}regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 :	�2head_7/kernel
:2head_7/bias
/
0
�1"
trackable_list_wrapper
/
0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
#:! 2conv_1/kernel
: 2conv_1/bias
#:! @2conv_2/kernel
:@2conv_2/bias
#:!@`2conv_3/kernel
:`2conv_3/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
 1
!2
"3
#4
$5
%6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layer_metrics
�metrics
 �layer_regularization_losses
�	variables
�trainable_variables
�layers
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 106}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "head_1_loss", "dtype": "float32", "config": {"name": "head_1_loss", "dtype": "float32"}, "shared_object_id": 107}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "head_2_loss", "dtype": "float32", "config": {"name": "head_2_loss", "dtype": "float32"}, "shared_object_id": 108}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "head_3_loss", "dtype": "float32", "config": {"name": "head_3_loss", "dtype": "float32"}, "shared_object_id": 109}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "head_4_loss", "dtype": "float32", "config": {"name": "head_4_loss", "dtype": "float32"}, "shared_object_id": 110}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "head_5_loss", "dtype": "float32", "config": {"name": "head_5_loss", "dtype": "float32"}, "shared_object_id": 111}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "head_6_loss", "dtype": "float32", "config": {"name": "head_6_loss", "dtype": "float32"}, "shared_object_id": 112}
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "head_7_loss", "dtype": "float32", "config": {"name": "head_7_loss", "dtype": "float32"}, "shared_object_id": 113}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "head_1_accuracy", "dtype": "float32", "config": {"name": "head_1_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 78}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "head_2_accuracy", "dtype": "float32", "config": {"name": "head_2_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 79}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "head_3_accuracy", "dtype": "float32", "config": {"name": "head_3_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 80}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "head_4_accuracy", "dtype": "float32", "config": {"name": "head_4_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 81}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "head_5_accuracy", "dtype": "float32", "config": {"name": "head_5_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 82}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "head_6_accuracy", "dtype": "float32", "config": {"name": "head_6_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 83}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "head_7_accuracy", "dtype": "float32", "config": {"name": "head_7_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 84}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
&:$
��2Adam/dens_1/kernel/m
:�2Adam/dens_1/bias/m
&:$
��2Adam/dens_2/kernel/m
:�2Adam/dens_2/bias/m
&:$
��2Adam/dens_3/kernel/m
:�2Adam/dens_3/bias/m
&:$
��2Adam/dens_4/kernel/m
:�2Adam/dens_4/bias/m
&:$
��2Adam/dens_5/kernel/m
:�2Adam/dens_5/bias/m
&:$
��2Adam/dens_6/kernel/m
:�2Adam/dens_6/bias/m
&:$
��2Adam/dens_7/kernel/m
:�2Adam/dens_7/bias/m
%:#	�2Adam/head_1/kernel/m
:2Adam/head_1/bias/m
%:#	�2Adam/head_2/kernel/m
:2Adam/head_2/bias/m
%:#	�2Adam/head_3/kernel/m
:2Adam/head_3/bias/m
%:#	�2Adam/head_4/kernel/m
:2Adam/head_4/bias/m
%:#	�2Adam/head_5/kernel/m
:2Adam/head_5/bias/m
%:#	�2Adam/head_6/kernel/m
:2Adam/head_6/bias/m
%:#	�2Adam/head_7/kernel/m
:2Adam/head_7/bias/m
(:& 2Adam/conv_1/kernel/m
: 2Adam/conv_1/bias/m
(:& @2Adam/conv_2/kernel/m
:@2Adam/conv_2/bias/m
(:&@`2Adam/conv_3/kernel/m
:`2Adam/conv_3/bias/m
&:$
��2Adam/dens_1/kernel/v
:�2Adam/dens_1/bias/v
&:$
��2Adam/dens_2/kernel/v
:�2Adam/dens_2/bias/v
&:$
��2Adam/dens_3/kernel/v
:�2Adam/dens_3/bias/v
&:$
��2Adam/dens_4/kernel/v
:�2Adam/dens_4/bias/v
&:$
��2Adam/dens_5/kernel/v
:�2Adam/dens_5/bias/v
&:$
��2Adam/dens_6/kernel/v
:�2Adam/dens_6/bias/v
&:$
��2Adam/dens_7/kernel/v
:�2Adam/dens_7/bias/v
%:#	�2Adam/head_1/kernel/v
:2Adam/head_1/bias/v
%:#	�2Adam/head_2/kernel/v
:2Adam/head_2/bias/v
%:#	�2Adam/head_3/kernel/v
:2Adam/head_3/bias/v
%:#	�2Adam/head_4/kernel/v
:2Adam/head_4/bias/v
%:#	�2Adam/head_5/kernel/v
:2Adam/head_5/bias/v
%:#	�2Adam/head_6/kernel/v
:2Adam/head_6/bias/v
%:#	�2Adam/head_7/kernel/v
:2Adam/head_7/bias/v
(:& 2Adam/conv_1/kernel/v
: 2Adam/conv_1/bias/v
(:& @2Adam/conv_2/kernel/v
:@2Adam/conv_2/bias/v
(:&@`2Adam/conv_3/kernel/v
:`2Adam/conv_3/bias/v
�2�
;__inference_multi-task_self-supervised_layer_call_fn_217322
;__inference_multi-task_self-supervised_layer_call_fn_218441
;__inference_multi-task_self-supervised_layer_call_fn_218532
;__inference_multi-task_self-supervised_layer_call_fn_217901�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_216405�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *���
���
&�#
input_1����������
&�#
input_2����������
&�#
input_3����������
&�#
input_4����������
&�#
input_5����������
&�#
input_6����������
&�#
input_7����������
�2�
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_218943
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_219501
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_218067
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_218233�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_trunk__layer_call_fn_216551
'__inference_trunk__layer_call_fn_219536
'__inference_trunk__layer_call_fn_219553
'__inference_trunk__layer_call_fn_216741�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_trunk__layer_call_and_return_conditional_losses_219614
B__inference_trunk__layer_call_and_return_conditional_losses_219696
B__inference_trunk__layer_call_and_return_conditional_losses_216781
B__inference_trunk__layer_call_and_return_conditional_losses_216821�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_global_max_pool__layer_call_fn_216856
1__inference_global_max_pool__layer_call_fn_219701
1__inference_global_max_pool__layer_call_fn_219706
1__inference_global_max_pool__layer_call_fn_216883�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_global_max_pool__layer_call_and_return_conditional_losses_219716
L__inference_global_max_pool__layer_call_and_return_conditional_losses_219726
L__inference_global_max_pool__layer_call_and_return_conditional_losses_216889
L__inference_global_max_pool__layer_call_and_return_conditional_losses_216895�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_dens_1_layer_call_fn_219735�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dens_1_layer_call_and_return_conditional_losses_219746�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dens_2_layer_call_fn_219755�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dens_2_layer_call_and_return_conditional_losses_219766�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dens_3_layer_call_fn_219775�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dens_3_layer_call_and_return_conditional_losses_219786�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dens_4_layer_call_fn_219795�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dens_4_layer_call_and_return_conditional_losses_219806�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dens_5_layer_call_fn_219815�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dens_5_layer_call_and_return_conditional_losses_219826�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dens_6_layer_call_fn_219835�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dens_6_layer_call_and_return_conditional_losses_219846�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dens_7_layer_call_fn_219855�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dens_7_layer_call_and_return_conditional_losses_219866�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_head_1_layer_call_fn_219875�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_head_1_layer_call_and_return_conditional_losses_219886�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_head_2_layer_call_fn_219895�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_head_2_layer_call_and_return_conditional_losses_219906�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_head_3_layer_call_fn_219915�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_head_3_layer_call_and_return_conditional_losses_219926�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_head_4_layer_call_fn_219935�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_head_4_layer_call_and_return_conditional_losses_219946�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_head_5_layer_call_fn_219955�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_head_5_layer_call_and_return_conditional_losses_219966�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_head_6_layer_call_fn_219975�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_head_6_layer_call_and_return_conditional_losses_219986�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_head_7_layer_call_fn_219995�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_head_7_layer_call_and_return_conditional_losses_220006�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_218350input_1input_2input_3input_4input_5input_6input_7"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv_1_layer_call_fn_220021�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv_1_layer_call_and_return_conditional_losses_220043�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_drop_1_layer_call_fn_220048
'__inference_drop_1_layer_call_fn_220053�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_drop_1_layer_call_and_return_conditional_losses_220058
B__inference_drop_1_layer_call_and_return_conditional_losses_220070�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_conv_2_layer_call_fn_220085�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv_2_layer_call_and_return_conditional_losses_220107�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_drop_2_layer_call_fn_220112
'__inference_drop_2_layer_call_fn_220117�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_drop_2_layer_call_and_return_conditional_losses_220122
B__inference_drop_2_layer_call_and_return_conditional_losses_220134�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_conv_3_layer_call_fn_220149�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv_3_layer_call_and_return_conditional_losses_220171�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_drop_3_layer_call_fn_220176
'__inference_drop_3_layer_call_fn_220181�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_drop_3_layer_call_and_return_conditional_losses_220186
B__inference_drop_3_layer_call_and_return_conditional_losses_220198�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference_loss_fn_0_220209�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_220220�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_220231�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
&__inference_pool__layer_call_fn_216836�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
A__inference_pool__layer_call_and_return_conditional_losses_216830�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
&__inference_flat__layer_call_fn_220236�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_flat__layer_call_and_return_conditional_losses_220242�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_216405�)������UVOPIJCD=>7812�yzstmnghab[\���
���
���
&�#
input_1����������
&�#
input_2����������
&�#
input_3����������
&�#
input_4����������
&�#
input_5����������
&�#
input_6����������
&�#
input_7����������
� "���
*
head_1 �
head_1���������
*
head_2 �
head_2���������
*
head_3 �
head_3���������
*
head_4 �
head_4���������
*
head_5 �
head_5���������
*
head_6 �
head_6���������
*
head_7 �
head_7����������
B__inference_conv_1_layer_call_and_return_conditional_losses_220043g��4�1
*�'
%�"
inputs����������
� ")�&
�
0���������i 
� �
'__inference_conv_1_layer_call_fn_220021Z��4�1
*�'
%�"
inputs����������
� "����������i �
B__inference_conv_2_layer_call_and_return_conditional_losses_220107f��3�0
)�&
$�!
inputs���������i 
� ")�&
�
0���������Z@
� �
'__inference_conv_2_layer_call_fn_220085Y��3�0
)�&
$�!
inputs���������i 
� "����������Z@�
B__inference_conv_3_layer_call_and_return_conditional_losses_220171f��3�0
)�&
$�!
inputs���������Z@
� ")�&
�
0���������S`
� �
'__inference_conv_3_layer_call_fn_220149Y��3�0
)�&
$�!
inputs���������Z@
� "����������S`�
B__inference_dens_1_layer_call_and_return_conditional_losses_219746^120�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dens_1_layer_call_fn_219735Q120�-
&�#
!�
inputs����������
� "������������
B__inference_dens_2_layer_call_and_return_conditional_losses_219766^780�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dens_2_layer_call_fn_219755Q780�-
&�#
!�
inputs����������
� "������������
B__inference_dens_3_layer_call_and_return_conditional_losses_219786^=>0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dens_3_layer_call_fn_219775Q=>0�-
&�#
!�
inputs����������
� "������������
B__inference_dens_4_layer_call_and_return_conditional_losses_219806^CD0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dens_4_layer_call_fn_219795QCD0�-
&�#
!�
inputs����������
� "������������
B__inference_dens_5_layer_call_and_return_conditional_losses_219826^IJ0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dens_5_layer_call_fn_219815QIJ0�-
&�#
!�
inputs����������
� "������������
B__inference_dens_6_layer_call_and_return_conditional_losses_219846^OP0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dens_6_layer_call_fn_219835QOP0�-
&�#
!�
inputs����������
� "������������
B__inference_dens_7_layer_call_and_return_conditional_losses_219866^UV0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dens_7_layer_call_fn_219855QUV0�-
&�#
!�
inputs����������
� "������������
B__inference_drop_1_layer_call_and_return_conditional_losses_220058d7�4
-�*
$�!
inputs���������i 
p 
� ")�&
�
0���������i 
� �
B__inference_drop_1_layer_call_and_return_conditional_losses_220070d7�4
-�*
$�!
inputs���������i 
p
� ")�&
�
0���������i 
� �
'__inference_drop_1_layer_call_fn_220048W7�4
-�*
$�!
inputs���������i 
p 
� "����������i �
'__inference_drop_1_layer_call_fn_220053W7�4
-�*
$�!
inputs���������i 
p
� "����������i �
B__inference_drop_2_layer_call_and_return_conditional_losses_220122d7�4
-�*
$�!
inputs���������Z@
p 
� ")�&
�
0���������Z@
� �
B__inference_drop_2_layer_call_and_return_conditional_losses_220134d7�4
-�*
$�!
inputs���������Z@
p
� ")�&
�
0���������Z@
� �
'__inference_drop_2_layer_call_fn_220112W7�4
-�*
$�!
inputs���������Z@
p 
� "����������Z@�
'__inference_drop_2_layer_call_fn_220117W7�4
-�*
$�!
inputs���������Z@
p
� "����������Z@�
B__inference_drop_3_layer_call_and_return_conditional_losses_220186d7�4
-�*
$�!
inputs���������S`
p 
� ")�&
�
0���������S`
� �
B__inference_drop_3_layer_call_and_return_conditional_losses_220198d7�4
-�*
$�!
inputs���������S`
p
� ")�&
�
0���������S`
� �
'__inference_drop_3_layer_call_fn_220176W7�4
-�*
$�!
inputs���������S`
p 
� "����������S`�
'__inference_drop_3_layer_call_fn_220181W7�4
-�*
$�!
inputs���������S`
p
� "����������S`�
A__inference_flat__layer_call_and_return_conditional_losses_220242]3�0
)�&
$�!
inputs���������)`
� "&�#
�
0����������
� z
&__inference_flat__layer_call_fn_220236P3�0
)�&
$�!
inputs���������)`
� "������������
L__inference_global_max_pool__layer_call_and_return_conditional_losses_216889f<�9
2�/
%�"
input_I���������S`
p 

 
� "&�#
�
0����������
� �
L__inference_global_max_pool__layer_call_and_return_conditional_losses_216895f<�9
2�/
%�"
input_I���������S`
p

 
� "&�#
�
0����������
� �
L__inference_global_max_pool__layer_call_and_return_conditional_losses_219716e;�8
1�.
$�!
inputs���������S`
p 

 
� "&�#
�
0����������
� �
L__inference_global_max_pool__layer_call_and_return_conditional_losses_219726e;�8
1�.
$�!
inputs���������S`
p

 
� "&�#
�
0����������
� �
1__inference_global_max_pool__layer_call_fn_216856Y<�9
2�/
%�"
input_I���������S`
p 

 
� "������������
1__inference_global_max_pool__layer_call_fn_216883Y<�9
2�/
%�"
input_I���������S`
p

 
� "������������
1__inference_global_max_pool__layer_call_fn_219701X;�8
1�.
$�!
inputs���������S`
p 

 
� "������������
1__inference_global_max_pool__layer_call_fn_219706X;�8
1�.
$�!
inputs���������S`
p

 
� "������������
B__inference_head_1_layer_call_and_return_conditional_losses_219886][\0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_head_1_layer_call_fn_219875P[\0�-
&�#
!�
inputs����������
� "�����������
B__inference_head_2_layer_call_and_return_conditional_losses_219906]ab0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_head_2_layer_call_fn_219895Pab0�-
&�#
!�
inputs����������
� "�����������
B__inference_head_3_layer_call_and_return_conditional_losses_219926]gh0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_head_3_layer_call_fn_219915Pgh0�-
&�#
!�
inputs����������
� "�����������
B__inference_head_4_layer_call_and_return_conditional_losses_219946]mn0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_head_4_layer_call_fn_219935Pmn0�-
&�#
!�
inputs����������
� "�����������
B__inference_head_5_layer_call_and_return_conditional_losses_219966]st0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_head_5_layer_call_fn_219955Pst0�-
&�#
!�
inputs����������
� "�����������
B__inference_head_6_layer_call_and_return_conditional_losses_219986]yz0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_head_6_layer_call_fn_219975Pyz0�-
&�#
!�
inputs����������
� "�����������
B__inference_head_7_layer_call_and_return_conditional_losses_220006^�0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� |
'__inference_head_7_layer_call_fn_219995Q�0�-
&�#
!�
inputs����������
� "����������<
__inference_loss_fn_0_220209��

� 
� "� <
__inference_loss_fn_1_220220��

� 
� "� <
__inference_loss_fn_2_220231��

� 
� "� �
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_218067�)������UVOPIJCD=>7812�yzstmnghab[\���
���
���
&�#
input_1����������
&�#
input_2����������
&�#
input_3����������
&�#
input_4����������
&�#
input_5����������
&�#
input_6����������
&�#
input_7����������
p 

 
� "���
���
�
0/0���������
�
0/1���������
�
0/2���������
�
0/3���������
�
0/4���������
�
0/5���������
�
0/6���������
� �
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_218233�)������UVOPIJCD=>7812�yzstmnghab[\���
���
���
&�#
input_1����������
&�#
input_2����������
&�#
input_3����������
&�#
input_4����������
&�#
input_5����������
&�#
input_6����������
&�#
input_7����������
p

 
� "���
���
�
0/0���������
�
0/1���������
�
0/2���������
�
0/3���������
�
0/4���������
�
0/5���������
�
0/6���������
� �
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_218943�)������UVOPIJCD=>7812�yzstmnghab[\���
���
���
'�$
inputs/0����������
'�$
inputs/1����������
'�$
inputs/2����������
'�$
inputs/3����������
'�$
inputs/4����������
'�$
inputs/5����������
'�$
inputs/6����������
p 

 
� "���
���
�
0/0���������
�
0/1���������
�
0/2���������
�
0/3���������
�
0/4���������
�
0/5���������
�
0/6���������
� �
V__inference_multi-task_self-supervised_layer_call_and_return_conditional_losses_219501�)������UVOPIJCD=>7812�yzstmnghab[\���
���
���
'�$
inputs/0����������
'�$
inputs/1����������
'�$
inputs/2����������
'�$
inputs/3����������
'�$
inputs/4����������
'�$
inputs/5����������
'�$
inputs/6����������
p

 
� "���
���
�
0/0���������
�
0/1���������
�
0/2���������
�
0/3���������
�
0/4���������
�
0/5���������
�
0/6���������
� �
;__inference_multi-task_self-supervised_layer_call_fn_217322�)������UVOPIJCD=>7812�yzstmnghab[\���
���
���
&�#
input_1����������
&�#
input_2����������
&�#
input_3����������
&�#
input_4����������
&�#
input_5����������
&�#
input_6����������
&�#
input_7����������
p 

 
� "���
�
0���������
�
1���������
�
2���������
�
3���������
�
4���������
�
5���������
�
6����������
;__inference_multi-task_self-supervised_layer_call_fn_217901�)������UVOPIJCD=>7812�yzstmnghab[\���
���
���
&�#
input_1����������
&�#
input_2����������
&�#
input_3����������
&�#
input_4����������
&�#
input_5����������
&�#
input_6����������
&�#
input_7����������
p

 
� "���
�
0���������
�
1���������
�
2���������
�
3���������
�
4���������
�
5���������
�
6����������
;__inference_multi-task_self-supervised_layer_call_fn_218441�)������UVOPIJCD=>7812�yzstmnghab[\���
���
���
'�$
inputs/0����������
'�$
inputs/1����������
'�$
inputs/2����������
'�$
inputs/3����������
'�$
inputs/4����������
'�$
inputs/5����������
'�$
inputs/6����������
p 

 
� "���
�
0���������
�
1���������
�
2���������
�
3���������
�
4���������
�
5���������
�
6����������
;__inference_multi-task_self-supervised_layer_call_fn_218532�)������UVOPIJCD=>7812�yzstmnghab[\���
���
���
'�$
inputs/0����������
'�$
inputs/1����������
'�$
inputs/2����������
'�$
inputs/3����������
'�$
inputs/4����������
'�$
inputs/5����������
'�$
inputs/6����������
p

 
� "���
�
0���������
�
1���������
�
2���������
�
3���������
�
4���������
�
5���������
�
6����������
A__inference_pool__layer_call_and_return_conditional_losses_216830�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
&__inference_pool__layer_call_fn_216836wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
$__inference_signature_wrapper_218350�)������UVOPIJCD=>7812�yzstmnghab[\���
� 
���
1
input_1&�#
input_1����������
1
input_2&�#
input_2����������
1
input_3&�#
input_3����������
1
input_4&�#
input_4����������
1
input_5&�#
input_5����������
1
input_6&�#
input_6����������
1
input_7&�#
input_7����������"���
*
head_1 �
head_1���������
*
head_2 �
head_2���������
*
head_3 �
head_3���������
*
head_4 �
head_4���������
*
head_5 �
head_5���������
*
head_6 �
head_6���������
*
head_7 �
head_7����������
B__inference_trunk__layer_call_and_return_conditional_losses_216781w������<�9
2�/
%�"
input_����������
p 

 
� ")�&
�
0���������S`
� �
B__inference_trunk__layer_call_and_return_conditional_losses_216821w������<�9
2�/
%�"
input_����������
p

 
� ")�&
�
0���������S`
� �
B__inference_trunk__layer_call_and_return_conditional_losses_219614w������<�9
2�/
%�"
inputs����������
p 

 
� ")�&
�
0���������S`
� �
B__inference_trunk__layer_call_and_return_conditional_losses_219696w������<�9
2�/
%�"
inputs����������
p

 
� ")�&
�
0���������S`
� �
'__inference_trunk__layer_call_fn_216551j������<�9
2�/
%�"
input_����������
p 

 
� "����������S`�
'__inference_trunk__layer_call_fn_216741j������<�9
2�/
%�"
input_����������
p

 
� "����������S`�
'__inference_trunk__layer_call_fn_219536j������<�9
2�/
%�"
inputs����������
p 

 
� "����������S`�
'__inference_trunk__layer_call_fn_219553j������<�9
2�/
%�"
inputs����������
p

 
� "����������S`