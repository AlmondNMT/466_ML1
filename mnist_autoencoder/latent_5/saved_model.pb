??	
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
?
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
?
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18??
?
%Adam/module_wrapper_14/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/module_wrapper_14/dense_5/bias/v
?
9Adam/module_wrapper_14/dense_5/bias/v/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper_14/dense_5/bias/v*
_output_shapes	
:?*
dtype0
?
'Adam/module_wrapper_14/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?*8
shared_name)'Adam/module_wrapper_14/dense_5/kernel/v
?
;Adam/module_wrapper_14/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_14/dense_5/kernel/v*
_output_shapes
:	
?*
dtype0
?
%Adam/module_wrapper_13/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%Adam/module_wrapper_13/dense_4/bias/v
?
9Adam/module_wrapper_13/dense_4/bias/v/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper_13/dense_4/bias/v*
_output_shapes
:
*
dtype0
?
'Adam/module_wrapper_13/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*8
shared_name)'Adam/module_wrapper_13/dense_4/kernel/v
?
;Adam/module_wrapper_13/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_13/dense_4/kernel/v*
_output_shapes

:
*
dtype0
?
%Adam/module_wrapper_12/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/module_wrapper_12/dense_3/bias/v
?
9Adam/module_wrapper_12/dense_3/bias/v/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper_12/dense_3/bias/v*
_output_shapes
:*
dtype0
?
'Adam/module_wrapper_12/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*8
shared_name)'Adam/module_wrapper_12/dense_3/kernel/v
?
;Adam/module_wrapper_12/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_12/dense_3/kernel/v*
_output_shapes
:	?*
dtype0
?
&Adam/module_wrapper_10/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*7
shared_name(&Adam/module_wrapper_10/conv2d_3/bias/v
?
:Adam/module_wrapper_10/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_10/conv2d_3/bias/v*
_output_shapes
:<*
dtype0
?
(Adam/module_wrapper_10/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@<*9
shared_name*(Adam/module_wrapper_10/conv2d_3/kernel/v
?
<Adam/module_wrapper_10/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/module_wrapper_10/conv2d_3/kernel/v*&
_output_shapes
:@<*
dtype0
?
%Adam/module_wrapper_8/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/module_wrapper_8/conv2d_2/bias/v
?
9Adam/module_wrapper_8/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper_8/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
?
'Adam/module_wrapper_8/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/module_wrapper_8/conv2d_2/kernel/v
?
;Adam/module_wrapper_8/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_8/conv2d_2/kernel/v*&
_output_shapes
:@*
dtype0
?
%Adam/module_wrapper_14/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/module_wrapper_14/dense_5/bias/m
?
9Adam/module_wrapper_14/dense_5/bias/m/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper_14/dense_5/bias/m*
_output_shapes	
:?*
dtype0
?
'Adam/module_wrapper_14/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?*8
shared_name)'Adam/module_wrapper_14/dense_5/kernel/m
?
;Adam/module_wrapper_14/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_14/dense_5/kernel/m*
_output_shapes
:	
?*
dtype0
?
%Adam/module_wrapper_13/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%Adam/module_wrapper_13/dense_4/bias/m
?
9Adam/module_wrapper_13/dense_4/bias/m/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper_13/dense_4/bias/m*
_output_shapes
:
*
dtype0
?
'Adam/module_wrapper_13/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*8
shared_name)'Adam/module_wrapper_13/dense_4/kernel/m
?
;Adam/module_wrapper_13/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_13/dense_4/kernel/m*
_output_shapes

:
*
dtype0
?
%Adam/module_wrapper_12/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%Adam/module_wrapper_12/dense_3/bias/m
?
9Adam/module_wrapper_12/dense_3/bias/m/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper_12/dense_3/bias/m*
_output_shapes
:*
dtype0
?
'Adam/module_wrapper_12/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*8
shared_name)'Adam/module_wrapper_12/dense_3/kernel/m
?
;Adam/module_wrapper_12/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_12/dense_3/kernel/m*
_output_shapes
:	?*
dtype0
?
&Adam/module_wrapper_10/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*7
shared_name(&Adam/module_wrapper_10/conv2d_3/bias/m
?
:Adam/module_wrapper_10/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_10/conv2d_3/bias/m*
_output_shapes
:<*
dtype0
?
(Adam/module_wrapper_10/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@<*9
shared_name*(Adam/module_wrapper_10/conv2d_3/kernel/m
?
<Adam/module_wrapper_10/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/module_wrapper_10/conv2d_3/kernel/m*&
_output_shapes
:@<*
dtype0
?
%Adam/module_wrapper_8/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%Adam/module_wrapper_8/conv2d_2/bias/m
?
9Adam/module_wrapper_8/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper_8/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
?
'Adam/module_wrapper_8/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/module_wrapper_8/conv2d_2/kernel/m
?
;Adam/module_wrapper_8/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_8/conv2d_2/kernel/m*&
_output_shapes
:@*
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
?
module_wrapper_14/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name module_wrapper_14/dense_5/bias
?
2module_wrapper_14/dense_5/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_14/dense_5/bias*
_output_shapes	
:?*
dtype0
?
 module_wrapper_14/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?*1
shared_name" module_wrapper_14/dense_5/kernel
?
4module_wrapper_14/dense_5/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_14/dense_5/kernel*
_output_shapes
:	
?*
dtype0
?
module_wrapper_13/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name module_wrapper_13/dense_4/bias
?
2module_wrapper_13/dense_4/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_13/dense_4/bias*
_output_shapes
:
*
dtype0
?
 module_wrapper_13/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*1
shared_name" module_wrapper_13/dense_4/kernel
?
4module_wrapper_13/dense_4/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_13/dense_4/kernel*
_output_shapes

:
*
dtype0
?
module_wrapper_12/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name module_wrapper_12/dense_3/bias
?
2module_wrapper_12/dense_3/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_12/dense_3/bias*
_output_shapes
:*
dtype0
?
 module_wrapper_12/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" module_wrapper_12/dense_3/kernel
?
4module_wrapper_12/dense_3/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_12/dense_3/kernel*
_output_shapes
:	?*
dtype0
?
module_wrapper_10/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*0
shared_name!module_wrapper_10/conv2d_3/bias
?
3module_wrapper_10/conv2d_3/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_10/conv2d_3/bias*
_output_shapes
:<*
dtype0
?
!module_wrapper_10/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@<*2
shared_name#!module_wrapper_10/conv2d_3/kernel
?
5module_wrapper_10/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp!module_wrapper_10/conv2d_3/kernel*&
_output_shapes
:@<*
dtype0
?
module_wrapper_8/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name module_wrapper_8/conv2d_2/bias
?
2module_wrapper_8/conv2d_2/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_8/conv2d_2/bias*
_output_shapes
:@*
dtype0
?
 module_wrapper_8/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" module_wrapper_8/conv2d_2/kernel
?
4module_wrapper_8/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp module_wrapper_8/conv2d_2/kernel*&
_output_shapes
:@*
dtype0

NoOpNoOp
?U
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?U
value?TB?T B?T
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

regularization_losses
	variables
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
_default_save_signature
__call__
	optimizer

signatures*
* 
?
regularization_losses
	variables
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
__call__

kernel
bias*
?
regularization_losses
	variables
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
 __call__* 
?
!regularization_losses
"	variables
#trainable_variables
$	keras_api
*%&call_and_return_all_conditional_losses
&__call__

'kernel
(bias*
?
)regularization_losses
*	variables
+trainable_variables
,	keras_api
*-&call_and_return_all_conditional_losses
.__call__* 
?
/regularization_losses
0	variables
1trainable_variables
2	keras_api
*3&call_and_return_all_conditional_losses
4__call__

5kernel
6bias*
?
7regularization_losses
8	variables
9trainable_variables
:	keras_api
*;&call_and_return_all_conditional_losses
<__call__

=kernel
>bias*
?
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
*C&call_and_return_all_conditional_losses
D__call__

Ekernel
Fbias*
?
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
*K&call_and_return_all_conditional_losses
L__call__* 
* 
J
0
1
'2
(3
54
65
=6
>7
E8
F9*
J
0
1
'2
(3
54
65
=6
>7
E8
F9*
?
Mmetrics

regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables

Qlayers
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Rtrace_0
Strace_1
Ttrace_2
Utrace_3* 

Vtrace_0* 
6
Wtrace_0
Xtrace_1
Ytrace_2
Ztrace_3* 
?
[iter

\beta_1

]beta_2
	^decay
_learning_ratem?m?'m?(m?5m?6m?=m?>m?Em?Fm?v?v?'v?(v?5v?6v?=v?>v?Ev?Fv?*

`serving_default* 
* 

0
1*

0
1*
?
ametrics
regularization_losses
bnon_trainable_variables
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables

elayers
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ftrace_0* 

gtrace_0* 
pj
VARIABLE_VALUE module_wrapper_8/conv2d_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEmodule_wrapper_8/conv2d_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
hmetrics
regularization_losses
inon_trainable_variables
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables

llayers
 __call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

mtrace_0* 

ntrace_0* 
* 

'0
(1*

'0
(1*
?
ometrics
!regularization_losses
pnon_trainable_variables
qlayer_regularization_losses
rlayer_metrics
"	variables
#trainable_variables

slayers
&__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

ttrace_0* 

utrace_0* 
qk
VARIABLE_VALUE!module_wrapper_10/conv2d_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUEmodule_wrapper_10/conv2d_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
vmetrics
)regularization_losses
wnon_trainable_variables
xlayer_regularization_losses
ylayer_metrics
*	variables
+trainable_variables

zlayers
.__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

{trace_0* 

|trace_0* 
* 

50
61*

50
61*
?
}metrics
/regularization_losses
~non_trainable_variables
layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
?layers
4__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
pj
VARIABLE_VALUE module_wrapper_12/dense_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEmodule_wrapper_12/dense_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

=0
>1*

=0
>1*
?
?metrics
7regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
?layers
<__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
pj
VARIABLE_VALUE module_wrapper_13/dense_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEmodule_wrapper_13/dense_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

E0
F1*

E0
F1*
?
?metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
?layers
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
pj
VARIABLE_VALUE module_wrapper_14/dense_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUEmodule_wrapper_14/dense_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?metrics
Gregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
?layers
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*
* 
* 
* 
C
0
1
2
3
4
5
6
7
	8*
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
??
VARIABLE_VALUE'Adam/module_wrapper_8/conv2d_2/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/module_wrapper_8/conv2d_2/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE(Adam/module_wrapper_10/conv2d_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/module_wrapper_10/conv2d_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/module_wrapper_12/dense_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/module_wrapper_12/dense_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/module_wrapper_13/dense_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/module_wrapper_13/dense_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/module_wrapper_14/dense_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/module_wrapper_14/dense_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/module_wrapper_8/conv2d_2/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/module_wrapper_8/conv2d_2/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE(Adam/module_wrapper_10/conv2d_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE&Adam/module_wrapper_10/conv2d_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/module_wrapper_12/dense_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/module_wrapper_12/dense_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/module_wrapper_13/dense_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/module_wrapper_13/dense_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE'Adam/module_wrapper_14/dense_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/module_wrapper_14/dense_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_input_2Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2 module_wrapper_8/conv2d_2/kernelmodule_wrapper_8/conv2d_2/bias!module_wrapper_10/conv2d_3/kernelmodule_wrapper_10/conv2d_3/bias module_wrapper_12/dense_3/kernelmodule_wrapper_12/dense_3/bias module_wrapper_13/dense_4/kernelmodule_wrapper_13/dense_4/bias module_wrapper_14/dense_5/kernelmodule_wrapper_14/dense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_16137
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4module_wrapper_8/conv2d_2/kernel/Read/ReadVariableOp2module_wrapper_8/conv2d_2/bias/Read/ReadVariableOp5module_wrapper_10/conv2d_3/kernel/Read/ReadVariableOp3module_wrapper_10/conv2d_3/bias/Read/ReadVariableOp4module_wrapper_12/dense_3/kernel/Read/ReadVariableOp2module_wrapper_12/dense_3/bias/Read/ReadVariableOp4module_wrapper_13/dense_4/kernel/Read/ReadVariableOp2module_wrapper_13/dense_4/bias/Read/ReadVariableOp4module_wrapper_14/dense_5/kernel/Read/ReadVariableOp2module_wrapper_14/dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp;Adam/module_wrapper_8/conv2d_2/kernel/m/Read/ReadVariableOp9Adam/module_wrapper_8/conv2d_2/bias/m/Read/ReadVariableOp<Adam/module_wrapper_10/conv2d_3/kernel/m/Read/ReadVariableOp:Adam/module_wrapper_10/conv2d_3/bias/m/Read/ReadVariableOp;Adam/module_wrapper_12/dense_3/kernel/m/Read/ReadVariableOp9Adam/module_wrapper_12/dense_3/bias/m/Read/ReadVariableOp;Adam/module_wrapper_13/dense_4/kernel/m/Read/ReadVariableOp9Adam/module_wrapper_13/dense_4/bias/m/Read/ReadVariableOp;Adam/module_wrapper_14/dense_5/kernel/m/Read/ReadVariableOp9Adam/module_wrapper_14/dense_5/bias/m/Read/ReadVariableOp;Adam/module_wrapper_8/conv2d_2/kernel/v/Read/ReadVariableOp9Adam/module_wrapper_8/conv2d_2/bias/v/Read/ReadVariableOp<Adam/module_wrapper_10/conv2d_3/kernel/v/Read/ReadVariableOp:Adam/module_wrapper_10/conv2d_3/bias/v/Read/ReadVariableOp;Adam/module_wrapper_12/dense_3/kernel/v/Read/ReadVariableOp9Adam/module_wrapper_12/dense_3/bias/v/Read/ReadVariableOp;Adam/module_wrapper_13/dense_4/kernel/v/Read/ReadVariableOp9Adam/module_wrapper_13/dense_4/bias/v/Read/ReadVariableOp;Adam/module_wrapper_14/dense_5/kernel/v/Read/ReadVariableOp9Adam/module_wrapper_14/dense_5/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_16561
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename module_wrapper_8/conv2d_2/kernelmodule_wrapper_8/conv2d_2/bias!module_wrapper_10/conv2d_3/kernelmodule_wrapper_10/conv2d_3/bias module_wrapper_12/dense_3/kernelmodule_wrapper_12/dense_3/bias module_wrapper_13/dense_4/kernelmodule_wrapper_13/dense_4/bias module_wrapper_14/dense_5/kernelmodule_wrapper_14/dense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount'Adam/module_wrapper_8/conv2d_2/kernel/m%Adam/module_wrapper_8/conv2d_2/bias/m(Adam/module_wrapper_10/conv2d_3/kernel/m&Adam/module_wrapper_10/conv2d_3/bias/m'Adam/module_wrapper_12/dense_3/kernel/m%Adam/module_wrapper_12/dense_3/bias/m'Adam/module_wrapper_13/dense_4/kernel/m%Adam/module_wrapper_13/dense_4/bias/m'Adam/module_wrapper_14/dense_5/kernel/m%Adam/module_wrapper_14/dense_5/bias/m'Adam/module_wrapper_8/conv2d_2/kernel/v%Adam/module_wrapper_8/conv2d_2/bias/v(Adam/module_wrapper_10/conv2d_3/kernel/v&Adam/module_wrapper_10/conv2d_3/bias/v'Adam/module_wrapper_12/dense_3/kernel/v%Adam/module_wrapper_12/dense_3/bias/v'Adam/module_wrapper_13/dense_4/kernel/v%Adam/module_wrapper_13/dense_4/bias/v'Adam/module_wrapper_14/dense_5/kernel/v%Adam/module_wrapper_14/dense_5/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_16688??
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_16382

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_model_1_layer_call_fn_15871
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@<
	unknown_2:<
	unknown_3:	?
	unknown_4:
	unknown_5:

	unknown_6:

	unknown_7:	
?
	unknown_8:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_15848w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?#
?
B__inference_model_1_layer_call_and_return_conditional_losses_16104
input_2(
conv2d_2_16075:@
conv2d_2_16077:@(
conv2d_3_16081:@<
conv2d_3_16083:< 
dense_3_16087:	?
dense_3_16089:
dense_4_16092:

dense_4_16094:
 
dense_5_16097:	
?
dense_5_16099:	?
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_16075conv2d_2_16077*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15748?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15724?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_16081conv2d_3_16083*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15766?
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_15778?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_16087dense_3_16089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15791?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_16092dense_4_16094*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15808?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_16097dense_5_16099*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_15825?
reshape_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_15845y
IdentityIdentity"reshape_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_16342

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_16362

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_dense_3_layer_call_fn_16351

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15791o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?;
?
B__inference_model_1_layer_call_and_return_conditional_losses_16239

inputsA
'conv2d_2_conv2d_readvariableop_resource:@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@<6
(conv2d_3_biasadd_readvariableop_resource:<9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:
5
'dense_4_biasadd_readvariableop_resource:
9
&dense_5_matmul_readvariableop_resource:	
?6
'dense_5_biasadd_readvariableop_resource:	?
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@<*
dtype0?
conv2d_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_1/ReshapeReshapeconv2d_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????g
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????R
reshape_1/ShapeShapedense_5/Sigmoid:y:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_1/ReshapeReshapedense_5/Sigmoid:y:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????q
IdentityIdentityreshape_1/Reshape:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15748

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_15845

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_3_layer_call_fn_16320

inputs!
unknown:@<
	unknown_0:<
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15766w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?#
?
B__inference_model_1_layer_call_and_return_conditional_losses_15992

inputs(
conv2d_2_15963:@
conv2d_2_15965:@(
conv2d_3_15969:@<
conv2d_3_15971:< 
dense_3_15975:	?
dense_3_15977:
dense_4_15980:

dense_4_15982:
 
dense_5_15985:	
?
dense_5_15987:	?
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_15963conv2d_2_15965*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15748?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15724?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_15969conv2d_3_15971*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15766?
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_15778?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_15975dense_3_15977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15791?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_15980dense_4_15982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15808?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_15985dense_5_15987*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_15825?
reshape_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_15845y
IdentityIdentity"reshape_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_16402

inputs1
matmul_readvariableop_resource:	
?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_15808

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_1_layer_call_fn_16336

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_15778a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?#
?
B__inference_model_1_layer_call_and_return_conditional_losses_15848

inputs(
conv2d_2_15749:@
conv2d_2_15751:@(
conv2d_3_15767:@<
conv2d_3_15769:< 
dense_3_15792:	?
dense_3_15794:
dense_4_15809:

dense_4_15811:
 
dense_5_15826:	
?
dense_5_15828:	?
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_15749conv2d_2_15751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15748?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15724?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_15767conv2d_3_15769*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15766?
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_15778?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_15792dense_3_15794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15791?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_15809dense_4_15811*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15808?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_15826dense_5_15828*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_15825?
reshape_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_15845y
IdentityIdentity"reshape_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?;
?
B__inference_model_1_layer_call_and_return_conditional_losses_16291

inputsA
'conv2d_2_conv2d_readvariableop_resource:@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@<6
(conv2d_3_biasadd_readvariableop_resource:<9
&dense_3_matmul_readvariableop_resource:	?5
'dense_3_biasadd_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:
5
'dense_4_biasadd_readvariableop_resource:
9
&dense_5_matmul_readvariableop_resource:	
?6
'dense_5_biasadd_readvariableop_resource:	?
identity??conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@<*
dtype0?
conv2d_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_1/ReshapeReshapeconv2d_3/Relu:activations:0flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????g
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????R
reshape_1/ShapeShapedense_5/Sigmoid:y:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_1/ReshapeReshapedense_5/Sigmoid:y:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????q
IdentityIdentityreshape_1/Reshape:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_16137
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@<
	unknown_2:<
	unknown_3:	?
	unknown_4:
	unknown_5:

	unknown_6:

	unknown_7:	
?
	unknown_8:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_15718w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?C
?
 __inference__wrapped_model_15718
input_2I
/model_1_conv2d_2_conv2d_readvariableop_resource:@>
0model_1_conv2d_2_biasadd_readvariableop_resource:@I
/model_1_conv2d_3_conv2d_readvariableop_resource:@<>
0model_1_conv2d_3_biasadd_readvariableop_resource:<A
.model_1_dense_3_matmul_readvariableop_resource:	?=
/model_1_dense_3_biasadd_readvariableop_resource:@
.model_1_dense_4_matmul_readvariableop_resource:
=
/model_1_dense_4_biasadd_readvariableop_resource:
A
.model_1_dense_5_matmul_readvariableop_resource:	
?>
/model_1_dense_5_biasadd_readvariableop_resource:	?
identity??'model_1/conv2d_2/BiasAdd/ReadVariableOp?&model_1/conv2d_2/Conv2D/ReadVariableOp?'model_1/conv2d_3/BiasAdd/ReadVariableOp?&model_1/conv2d_3/Conv2D/ReadVariableOp?&model_1/dense_3/BiasAdd/ReadVariableOp?%model_1/dense_3/MatMul/ReadVariableOp?&model_1/dense_4/BiasAdd/ReadVariableOp?%model_1/dense_4/MatMul/ReadVariableOp?&model_1/dense_5/BiasAdd/ReadVariableOp?%model_1/dense_5/MatMul/ReadVariableOp?
&model_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
model_1/conv2d_2/Conv2DConv2Dinput_2.model_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
'model_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_1/conv2d_2/BiasAddBiasAdd model_1/conv2d_2/Conv2D:output:0/model_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@z
model_1/conv2d_2/ReluRelu!model_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
model_1/max_pooling2d_1/MaxPoolMaxPool#model_1/conv2d_2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
&model_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@<*
dtype0?
model_1/conv2d_3/Conv2DConv2D(model_1/max_pooling2d_1/MaxPool:output:0.model_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
?
'model_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0?
model_1/conv2d_3/BiasAddBiasAdd model_1/conv2d_3/Conv2D:output:0/model_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<z
model_1/conv2d_3/ReluRelu!model_1/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????<h
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
model_1/flatten_1/ReshapeReshape#model_1/conv2d_3/Relu:activations:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:???????????
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/dense_3/MatMulMatMul"model_1/flatten_1/Reshape:output:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0?
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
p
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
model_1/dense_5/MatMulMatMul"model_1/dense_4/Relu:activations:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????w
model_1/dense_5/SigmoidSigmoid model_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????b
model_1/reshape_1/ShapeShapemodel_1/dense_5/Sigmoid:y:0*
T0*
_output_shapes
:o
%model_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model_1/reshape_1/strided_sliceStridedSlice model_1/reshape_1/Shape:output:0.model_1/reshape_1/strided_slice/stack:output:00model_1/reshape_1/strided_slice/stack_1:output:00model_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!model_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
model_1/reshape_1/Reshape/shapePack(model_1/reshape_1/strided_slice:output:0*model_1/reshape_1/Reshape/shape/1:output:0*model_1/reshape_1/Reshape/shape/2:output:0*model_1/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
model_1/reshape_1/ReshapeReshapemodel_1/dense_5/Sigmoid:y:0(model_1/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????y
IdentityIdentity"model_1/reshape_1/Reshape:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^model_1/conv2d_2/BiasAdd/ReadVariableOp'^model_1/conv2d_2/Conv2D/ReadVariableOp(^model_1/conv2d_3/BiasAdd/ReadVariableOp'^model_1/conv2d_3/Conv2D/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2R
'model_1/conv2d_2/BiasAdd/ReadVariableOp'model_1/conv2d_2/BiasAdd/ReadVariableOp2P
&model_1/conv2d_2/Conv2D/ReadVariableOp&model_1/conv2d_2/Conv2D/ReadVariableOp2R
'model_1/conv2d_3/BiasAdd/ReadVariableOp'model_1/conv2d_3/BiasAdd/ReadVariableOp2P
&model_1/conv2d_3/Conv2D/ReadVariableOp&model_1/conv2d_3/Conv2D/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15766

inputs8
conv2d_readvariableop_resource:@<-
biasadd_readvariableop_resource:<
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@<*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
E
)__inference_reshape_1_layer_call_fn_16407

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_15845h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
'__inference_model_1_layer_call_fn_16162

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@<
	unknown_2:<
	unknown_3:	?
	unknown_4:
	unknown_5:

	unknown_6:

	unknown_7:	
?
	unknown_8:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_15848w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_4_layer_call_fn_16371

inputs
unknown:

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15808o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15724

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_15778

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????<:W S
/
_output_shapes
:?????????<
 
_user_specified_nameinputs
?

?
'__inference_model_1_layer_call_fn_16187

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@<
	unknown_2:<
	unknown_3:	?
	unknown_4:
	unknown_5:

	unknown_6:

	unknown_7:	
?
	unknown_8:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_15992w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?#
?
B__inference_model_1_layer_call_and_return_conditional_losses_16072
input_2(
conv2d_2_16043:@
conv2d_2_16045:@(
conv2d_3_16049:@<
conv2d_3_16051:< 
dense_3_16055:	?
dense_3_16057:
dense_4_16060:

dense_4_16062:
 
dense_5_16065:	
?
dense_5_16067:	?
identity?? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_2_16043conv2d_2_16045*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15748?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15724?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_16049conv2d_3_16051*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15766?
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_15778?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_16055dense_3_16057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_15791?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_16060dense_4_16062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_15808?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_16065dense_5_16067*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_15825?
reshape_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_15845y
IdentityIdentity"reshape_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2
?Y
?
__inference__traced_save_16561
file_prefix?
;savev2_module_wrapper_8_conv2d_2_kernel_read_readvariableop=
9savev2_module_wrapper_8_conv2d_2_bias_read_readvariableop@
<savev2_module_wrapper_10_conv2d_3_kernel_read_readvariableop>
:savev2_module_wrapper_10_conv2d_3_bias_read_readvariableop?
;savev2_module_wrapper_12_dense_3_kernel_read_readvariableop=
9savev2_module_wrapper_12_dense_3_bias_read_readvariableop?
;savev2_module_wrapper_13_dense_4_kernel_read_readvariableop=
9savev2_module_wrapper_13_dense_4_bias_read_readvariableop?
;savev2_module_wrapper_14_dense_5_kernel_read_readvariableop=
9savev2_module_wrapper_14_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopF
Bsavev2_adam_module_wrapper_8_conv2d_2_kernel_m_read_readvariableopD
@savev2_adam_module_wrapper_8_conv2d_2_bias_m_read_readvariableopG
Csavev2_adam_module_wrapper_10_conv2d_3_kernel_m_read_readvariableopE
Asavev2_adam_module_wrapper_10_conv2d_3_bias_m_read_readvariableopF
Bsavev2_adam_module_wrapper_12_dense_3_kernel_m_read_readvariableopD
@savev2_adam_module_wrapper_12_dense_3_bias_m_read_readvariableopF
Bsavev2_adam_module_wrapper_13_dense_4_kernel_m_read_readvariableopD
@savev2_adam_module_wrapper_13_dense_4_bias_m_read_readvariableopF
Bsavev2_adam_module_wrapper_14_dense_5_kernel_m_read_readvariableopD
@savev2_adam_module_wrapper_14_dense_5_bias_m_read_readvariableopF
Bsavev2_adam_module_wrapper_8_conv2d_2_kernel_v_read_readvariableopD
@savev2_adam_module_wrapper_8_conv2d_2_bias_v_read_readvariableopG
Csavev2_adam_module_wrapper_10_conv2d_3_kernel_v_read_readvariableopE
Asavev2_adam_module_wrapper_10_conv2d_3_bias_v_read_readvariableopF
Bsavev2_adam_module_wrapper_12_dense_3_kernel_v_read_readvariableopD
@savev2_adam_module_wrapper_12_dense_3_bias_v_read_readvariableopF
Bsavev2_adam_module_wrapper_13_dense_4_kernel_v_read_readvariableopD
@savev2_adam_module_wrapper_13_dense_4_bias_v_read_readvariableopF
Bsavev2_adam_module_wrapper_14_dense_5_kernel_v_read_readvariableopD
@savev2_adam_module_wrapper_14_dense_5_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_module_wrapper_8_conv2d_2_kernel_read_readvariableop9savev2_module_wrapper_8_conv2d_2_bias_read_readvariableop<savev2_module_wrapper_10_conv2d_3_kernel_read_readvariableop:savev2_module_wrapper_10_conv2d_3_bias_read_readvariableop;savev2_module_wrapper_12_dense_3_kernel_read_readvariableop9savev2_module_wrapper_12_dense_3_bias_read_readvariableop;savev2_module_wrapper_13_dense_4_kernel_read_readvariableop9savev2_module_wrapper_13_dense_4_bias_read_readvariableop;savev2_module_wrapper_14_dense_5_kernel_read_readvariableop9savev2_module_wrapper_14_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopBsavev2_adam_module_wrapper_8_conv2d_2_kernel_m_read_readvariableop@savev2_adam_module_wrapper_8_conv2d_2_bias_m_read_readvariableopCsavev2_adam_module_wrapper_10_conv2d_3_kernel_m_read_readvariableopAsavev2_adam_module_wrapper_10_conv2d_3_bias_m_read_readvariableopBsavev2_adam_module_wrapper_12_dense_3_kernel_m_read_readvariableop@savev2_adam_module_wrapper_12_dense_3_bias_m_read_readvariableopBsavev2_adam_module_wrapper_13_dense_4_kernel_m_read_readvariableop@savev2_adam_module_wrapper_13_dense_4_bias_m_read_readvariableopBsavev2_adam_module_wrapper_14_dense_5_kernel_m_read_readvariableop@savev2_adam_module_wrapper_14_dense_5_bias_m_read_readvariableopBsavev2_adam_module_wrapper_8_conv2d_2_kernel_v_read_readvariableop@savev2_adam_module_wrapper_8_conv2d_2_bias_v_read_readvariableopCsavev2_adam_module_wrapper_10_conv2d_3_kernel_v_read_readvariableopAsavev2_adam_module_wrapper_10_conv2d_3_bias_v_read_readvariableopBsavev2_adam_module_wrapper_12_dense_3_kernel_v_read_readvariableop@savev2_adam_module_wrapper_12_dense_3_bias_v_read_readvariableopBsavev2_adam_module_wrapper_13_dense_4_kernel_v_read_readvariableop@savev2_adam_module_wrapper_13_dense_4_bias_v_read_readvariableopBsavev2_adam_module_wrapper_14_dense_5_kernel_v_read_readvariableop@savev2_adam_module_wrapper_14_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@<:<:	?::
:
:	
?:?: : : : : : : : : :@:@:@<:<:	?::
:
:	
?:?:@:@:@<:<:	?::
:
:	
?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@<: 

_output_shapes
:<:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:%	!

_output_shapes
:	
?:!


_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@<: 

_output_shapes
:<:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:%!

_output_shapes
:	
?:!

_output_shapes	
:?:,(
&
_output_shapes
:@: 

_output_shapes
:@:, (
&
_output_shapes
:@<: !

_output_shapes
:<:%"!

_output_shapes
:	?: #

_output_shapes
::$$ 

_output_shapes

:
: %

_output_shapes
:
:%&!

_output_shapes
:	
?:!'

_output_shapes	
:?:(

_output_shapes
: 
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_15825

inputs1
matmul_readvariableop_resource:	
?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_16331

inputs8
conv2d_readvariableop_resource:@<-
biasadd_readvariableop_resource:<
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@<*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????<X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????<i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????<w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_15791

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_16311

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_5_layer_call_fn_16391

inputs
unknown:	
?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_15825p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_15730

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15724?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_16421

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_16688
file_prefixK
1assignvariableop_module_wrapper_8_conv2d_2_kernel:@?
1assignvariableop_1_module_wrapper_8_conv2d_2_bias:@N
4assignvariableop_2_module_wrapper_10_conv2d_3_kernel:@<@
2assignvariableop_3_module_wrapper_10_conv2d_3_bias:<F
3assignvariableop_4_module_wrapper_12_dense_3_kernel:	??
1assignvariableop_5_module_wrapper_12_dense_3_bias:E
3assignvariableop_6_module_wrapper_13_dense_4_kernel:
?
1assignvariableop_7_module_wrapper_13_dense_4_bias:
F
3assignvariableop_8_module_wrapper_14_dense_5_kernel:	
?@
1assignvariableop_9_module_wrapper_14_dense_5_bias:	?'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: U
;assignvariableop_19_adam_module_wrapper_8_conv2d_2_kernel_m:@G
9assignvariableop_20_adam_module_wrapper_8_conv2d_2_bias_m:@V
<assignvariableop_21_adam_module_wrapper_10_conv2d_3_kernel_m:@<H
:assignvariableop_22_adam_module_wrapper_10_conv2d_3_bias_m:<N
;assignvariableop_23_adam_module_wrapper_12_dense_3_kernel_m:	?G
9assignvariableop_24_adam_module_wrapper_12_dense_3_bias_m:M
;assignvariableop_25_adam_module_wrapper_13_dense_4_kernel_m:
G
9assignvariableop_26_adam_module_wrapper_13_dense_4_bias_m:
N
;assignvariableop_27_adam_module_wrapper_14_dense_5_kernel_m:	
?H
9assignvariableop_28_adam_module_wrapper_14_dense_5_bias_m:	?U
;assignvariableop_29_adam_module_wrapper_8_conv2d_2_kernel_v:@G
9assignvariableop_30_adam_module_wrapper_8_conv2d_2_bias_v:@V
<assignvariableop_31_adam_module_wrapper_10_conv2d_3_kernel_v:@<H
:assignvariableop_32_adam_module_wrapper_10_conv2d_3_bias_v:<N
;assignvariableop_33_adam_module_wrapper_12_dense_3_kernel_v:	?G
9assignvariableop_34_adam_module_wrapper_12_dense_3_bias_v:M
;assignvariableop_35_adam_module_wrapper_13_dense_4_kernel_v:
G
9assignvariableop_36_adam_module_wrapper_13_dense_4_bias_v:
N
;assignvariableop_37_adam_module_wrapper_14_dense_5_kernel_v:	
?H
9assignvariableop_38_adam_module_wrapper_14_dense_5_bias_v:	?
identity_40??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp1assignvariableop_module_wrapper_8_conv2d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp1assignvariableop_1_module_wrapper_8_conv2d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp4assignvariableop_2_module_wrapper_10_conv2d_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp2assignvariableop_3_module_wrapper_10_conv2d_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp3assignvariableop_4_module_wrapper_12_dense_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp1assignvariableop_5_module_wrapper_12_dense_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp3assignvariableop_6_module_wrapper_13_dense_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp1assignvariableop_7_module_wrapper_13_dense_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp3assignvariableop_8_module_wrapper_14_dense_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp1assignvariableop_9_module_wrapper_14_dense_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp;assignvariableop_19_adam_module_wrapper_8_conv2d_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp9assignvariableop_20_adam_module_wrapper_8_conv2d_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp<assignvariableop_21_adam_module_wrapper_10_conv2d_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp:assignvariableop_22_adam_module_wrapper_10_conv2d_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp;assignvariableop_23_adam_module_wrapper_12_dense_3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp9assignvariableop_24_adam_module_wrapper_12_dense_3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp;assignvariableop_25_adam_module_wrapper_13_dense_4_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp9assignvariableop_26_adam_module_wrapper_13_dense_4_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp;assignvariableop_27_adam_module_wrapper_14_dense_5_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp9assignvariableop_28_adam_module_wrapper_14_dense_5_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp;assignvariableop_29_adam_module_wrapper_8_conv2d_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp9assignvariableop_30_adam_module_wrapper_8_conv2d_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp<assignvariableop_31_adam_module_wrapper_10_conv2d_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp:assignvariableop_32_adam_module_wrapper_10_conv2d_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp;assignvariableop_33_adam_module_wrapper_12_dense_3_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp9assignvariableop_34_adam_module_wrapper_12_dense_3_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp;assignvariableop_35_adam_module_wrapper_13_dense_4_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp9assignvariableop_36_adam_module_wrapper_13_dense_4_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp;assignvariableop_37_adam_module_wrapper_14_dense_5_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp9assignvariableop_38_adam_module_wrapper_14_dense_5_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
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
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
(__inference_conv2d_2_layer_call_fn_16300

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15748w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_model_1_layer_call_fn_16040
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@<
	unknown_2:<
	unknown_3:	?
	unknown_4:
	unknown_5:

	unknown_6:

	unknown_7:	
?
	unknown_8:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_15992w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_2"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_28
serving_default_input_2:0?????????E
	reshape_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

regularization_losses
	variables
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
_default_save_signature
__call__
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?
regularization_losses
	variables
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
__call__

kernel
bias"
_tf_keras_layer
?
regularization_losses
	variables
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
 __call__"
_tf_keras_layer
?
!regularization_losses
"	variables
#trainable_variables
$	keras_api
*%&call_and_return_all_conditional_losses
&__call__

'kernel
(bias"
_tf_keras_layer
?
)regularization_losses
*	variables
+trainable_variables
,	keras_api
*-&call_and_return_all_conditional_losses
.__call__"
_tf_keras_layer
?
/regularization_losses
0	variables
1trainable_variables
2	keras_api
*3&call_and_return_all_conditional_losses
4__call__

5kernel
6bias"
_tf_keras_layer
?
7regularization_losses
8	variables
9trainable_variables
:	keras_api
*;&call_and_return_all_conditional_losses
<__call__

=kernel
>bias"
_tf_keras_layer
?
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
*C&call_and_return_all_conditional_losses
D__call__

Ekernel
Fbias"
_tf_keras_layer
?
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
*K&call_and_return_all_conditional_losses
L__call__"
_tf_keras_layer
 "
trackable_list_wrapper
f
0
1
'2
(3
54
65
=6
>7
E8
F9"
trackable_list_wrapper
f
0
1
'2
(3
54
65
=6
>7
E8
F9"
trackable_list_wrapper
?
Mmetrics

regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables

Qlayers
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Rtrace_0
Strace_1
Ttrace_2
Utrace_32?
B__inference_model_1_layer_call_and_return_conditional_losses_16239
B__inference_model_1_layer_call_and_return_conditional_losses_16291
B__inference_model_1_layer_call_and_return_conditional_losses_16072
B__inference_model_1_layer_call_and_return_conditional_losses_16104?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zRtrace_0zStrace_1zTtrace_2zUtrace_3
?
Vtrace_02?
 __inference__wrapped_model_15718?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_2?????????zVtrace_0
?
Wtrace_0
Xtrace_1
Ytrace_2
Ztrace_32?
'__inference_model_1_layer_call_fn_15871
'__inference_model_1_layer_call_fn_16162
'__inference_model_1_layer_call_fn_16187
'__inference_model_1_layer_call_fn_16040?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zWtrace_0zXtrace_1zYtrace_2zZtrace_3
?
[iter

\beta_1

]beta_2
	^decay
_learning_ratem?m?'m?(m?5m?6m?=m?>m?Em?Fm?v?v?'v?(v?5v?6v?=v?>v?Ev?Fv?"
tf_deprecated_optimizer
,
`serving_default"
signature_map
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
ametrics
regularization_losses
bnon_trainable_variables
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables

elayers
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
ftrace_02?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_16311?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zftrace_0
?
gtrace_02?
(__inference_conv2d_2_layer_call_fn_16300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zgtrace_0
::8@2 module_wrapper_8/conv2d_2/kernel
,:*@2module_wrapper_8/conv2d_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hmetrics
regularization_losses
inon_trainable_variables
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables

llayers
 __call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
mtrace_02?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15724?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????zmtrace_0
?
ntrace_02?
/__inference_max_pooling2d_1_layer_call_fn_15730?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????zntrace_0
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
ometrics
!regularization_losses
pnon_trainable_variables
qlayer_regularization_losses
rlayer_metrics
"	variables
#trainable_variables

slayers
&__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
?
ttrace_02?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_16331?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zttrace_0
?
utrace_02?
(__inference_conv2d_3_layer_call_fn_16320?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zutrace_0
;:9@<2!module_wrapper_10/conv2d_3/kernel
-:+<2module_wrapper_10/conv2d_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
vmetrics
)regularization_losses
wnon_trainable_variables
xlayer_regularization_losses
ylayer_metrics
*	variables
+trainable_variables

zlayers
.__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?
{trace_02?
D__inference_flatten_1_layer_call_and_return_conditional_losses_16342?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z{trace_0
?
|trace_02?
)__inference_flatten_1_layer_call_fn_16336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z|trace_0
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
}metrics
/regularization_losses
~non_trainable_variables
layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
?layers
4__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
B__inference_dense_3_layer_call_and_return_conditional_losses_16362?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
'__inference_dense_3_layer_call_fn_16351?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
3:1	?2 module_wrapper_12/dense_3/kernel
,:*2module_wrapper_12/dense_3/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?metrics
7regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
?layers
<__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
B__inference_dense_4_layer_call_and_return_conditional_losses_16382?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
'__inference_dense_4_layer_call_fn_16371?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
2:0
2 module_wrapper_13/dense_4/kernel
,:*
2module_wrapper_13/dense_4/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
?metrics
?regularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
?layers
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
B__inference_dense_5_layer_call_and_return_conditional_losses_16402?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
'__inference_dense_5_layer_call_fn_16391?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
3:1	
?2 module_wrapper_14/dense_5/kernel
-:+?2module_wrapper_14/dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
Gregularization_losses
?non_trainable_variables
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
?layers
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
D__inference_reshape_1_layer_call_and_return_conditional_losses_16421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
)__inference_reshape_1_layer_call_fn_16407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_16239inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_16291inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_16072input_2"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_1_layer_call_and_return_conditional_losses_16104input_2"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_15718input_2"?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_2?????????
?B?
'__inference_model_1_layer_call_fn_15871input_2"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_model_1_layer_call_fn_16162inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_model_1_layer_call_fn_16187inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_model_1_layer_call_fn_16040input_2"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
#__inference_signature_wrapper_16137input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_16311inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_conv2d_2_layer_call_fn_16300inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15724inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
/__inference_max_pooling2d_1_layer_call_fn_15730inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
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
?B?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_16331inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_conv2d_3_layer_call_fn_16320inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
D__inference_flatten_1_layer_call_and_return_conditional_losses_16342inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
)__inference_flatten_1_layer_call_fn_16336inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
B__inference_dense_3_layer_call_and_return_conditional_losses_16362inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_dense_3_layer_call_fn_16351inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
B__inference_dense_4_layer_call_and_return_conditional_losses_16382inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_dense_4_layer_call_fn_16371inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
B__inference_dense_5_layer_call_and_return_conditional_losses_16402inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_dense_5_layer_call_fn_16391inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
D__inference_reshape_1_layer_call_and_return_conditional_losses_16421inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
)__inference_reshape_1_layer_call_fn_16407inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
?:=@2'Adam/module_wrapper_8/conv2d_2/kernel/m
1:/@2%Adam/module_wrapper_8/conv2d_2/bias/m
@:>@<2(Adam/module_wrapper_10/conv2d_3/kernel/m
2:0<2&Adam/module_wrapper_10/conv2d_3/bias/m
8:6	?2'Adam/module_wrapper_12/dense_3/kernel/m
1:/2%Adam/module_wrapper_12/dense_3/bias/m
7:5
2'Adam/module_wrapper_13/dense_4/kernel/m
1:/
2%Adam/module_wrapper_13/dense_4/bias/m
8:6	
?2'Adam/module_wrapper_14/dense_5/kernel/m
2:0?2%Adam/module_wrapper_14/dense_5/bias/m
?:=@2'Adam/module_wrapper_8/conv2d_2/kernel/v
1:/@2%Adam/module_wrapper_8/conv2d_2/bias/v
@:>@<2(Adam/module_wrapper_10/conv2d_3/kernel/v
2:0<2&Adam/module_wrapper_10/conv2d_3/bias/v
8:6	?2'Adam/module_wrapper_12/dense_3/kernel/v
1:/2%Adam/module_wrapper_12/dense_3/bias/v
7:5
2'Adam/module_wrapper_13/dense_4/kernel/v
1:/
2%Adam/module_wrapper_13/dense_4/bias/v
8:6	
?2'Adam/module_wrapper_14/dense_5/kernel/v
2:0?2%Adam/module_wrapper_14/dense_5/bias/v?
 __inference__wrapped_model_15718?
'(56=>EF8?5
.?+
)?&
input_2?????????
? "=?:
8
	reshape_1+?(
	reshape_1??????????
C__inference_conv2d_2_layer_call_and_return_conditional_losses_16311l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_2_layer_call_fn_16300_7?4
-?*
(?%
inputs?????????
? " ??????????@?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_16331l'(7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????<
? ?
(__inference_conv2d_3_layer_call_fn_16320_'(7?4
-?*
(?%
inputs?????????@
? " ??????????<?
B__inference_dense_3_layer_call_and_return_conditional_losses_16362]560?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_3_layer_call_fn_16351P560?-
&?#
!?
inputs??????????
? "???????????
B__inference_dense_4_layer_call_and_return_conditional_losses_16382\=>/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? z
'__inference_dense_4_layer_call_fn_16371O=>/?,
%?"
 ?
inputs?????????
? "??????????
?
B__inference_dense_5_layer_call_and_return_conditional_losses_16402]EF/?,
%?"
 ?
inputs?????????

? "&?#
?
0??????????
? {
'__inference_dense_5_layer_call_fn_16391PEF/?,
%?"
 ?
inputs?????????

? "????????????
D__inference_flatten_1_layer_call_and_return_conditional_losses_16342a7?4
-?*
(?%
inputs?????????<
? "&?#
?
0??????????
? ?
)__inference_flatten_1_layer_call_fn_16336T7?4
-?*
(?%
inputs?????????<
? "????????????
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15724?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_15730?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_1_layer_call_and_return_conditional_losses_16072}
'(56=>EF@?=
6?3
)?&
input_2?????????
p 

 
? "-?*
#? 
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_16104}
'(56=>EF@?=
6?3
)?&
input_2?????????
p

 
? "-?*
#? 
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_16239|
'(56=>EF??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_16291|
'(56=>EF??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
'__inference_model_1_layer_call_fn_15871p
'(56=>EF@?=
6?3
)?&
input_2?????????
p 

 
? " ???????????
'__inference_model_1_layer_call_fn_16040p
'(56=>EF@?=
6?3
)?&
input_2?????????
p

 
? " ???????????
'__inference_model_1_layer_call_fn_16162o
'(56=>EF??<
5?2
(?%
inputs?????????
p 

 
? " ???????????
'__inference_model_1_layer_call_fn_16187o
'(56=>EF??<
5?2
(?%
inputs?????????
p

 
? " ???????????
D__inference_reshape_1_layer_call_and_return_conditional_losses_16421a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_1_layer_call_fn_16407T0?-
&?#
!?
inputs??????????
? " ???????????
#__inference_signature_wrapper_16137?
'(56=>EFC?@
? 
9?6
4
input_2)?&
input_2?????????"=?:
8
	reshape_1+?(
	reshape_1?????????