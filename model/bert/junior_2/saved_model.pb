??1
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint?
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
h
BatchMatMul
x"T
y"T
output"T"
Ttype:
	2"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	
?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
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
6
Pow
x"T
y"T
z"T"
Ttype:

2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
.
Rsqrt
x"T
y"T"
Ttype:

2
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
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.12.02v1.12.0-0-ga6d8ffae09??+
\
	input_idsPlaceholder*
dtype0	*
_output_shapes
:	?*
shape:	?
]

input_maskPlaceholder*
dtype0	*
_output_shapes
:	?*
shape:	?
^
segment_idsPlaceholder*
shape:	?*
dtype0	*
_output_shapes
:	?
M
valsPlaceholder*
_output_shapes
:*
shape:*
dtype0
q
bert/embeddings/ExpandDims/dimConst*
_output_shapes
:*
valueB:
?????????*
dtype0
?
bert/embeddings/ExpandDims
ExpandDims	input_idsbert/embeddings/ExpandDims/dim*

Tdim0*
T0	*#
_output_shapes
:?
?
Bbert/embeddings/word_embeddings/Initializer/truncated_normal/shapeConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
valueB":w     *
dtype0*
_output_shapes
:
?
Abert/embeddings/word_embeddings/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *2
_class(
&$loc:@bert/embeddings/word_embeddings*
valueB
 *    
?
Cbert/embeddings/word_embeddings/Initializer/truncated_normal/stddevConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
Lbert/embeddings/word_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalBbert/embeddings/word_embeddings/Initializer/truncated_normal/shape*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*
seed2 *
dtype0*!
_output_shapes
:???*

seed 
?
@bert/embeddings/word_embeddings/Initializer/truncated_normal/mulMulLbert/embeddings/word_embeddings/Initializer/truncated_normal/TruncatedNormalCbert/embeddings/word_embeddings/Initializer/truncated_normal/stddev*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:???*
T0
?
<bert/embeddings/word_embeddings/Initializer/truncated_normalAdd@bert/embeddings/word_embeddings/Initializer/truncated_normal/mulAbert/embeddings/word_embeddings/Initializer/truncated_normal/mean*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:???
?
bert/embeddings/word_embeddings
VariableV2*2
_class(
&$loc:@bert/embeddings/word_embeddings*
	container *
shape:???*
dtype0*!
_output_shapes
:???*
shared_name 
?
&bert/embeddings/word_embeddings/AssignAssignbert/embeddings/word_embeddings<bert/embeddings/word_embeddings/Initializer/truncated_normal*
use_locking(*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*
validate_shape(*!
_output_shapes
:???
?
$bert/embeddings/word_embeddings/readIdentitybert/embeddings/word_embeddings*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:???
p
bert/embeddings/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
bert/embeddings/ReshapeReshapebert/embeddings/ExpandDimsbert/embeddings/Reshape/shape*
T0	*
Tshape0*
_output_shapes	
:?
_
bert/embeddings/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
bert/embeddings/GatherV2GatherV2$bert/embeddings/word_embeddings/readbert/embeddings/Reshapebert/embeddings/GatherV2/axis* 
_output_shapes
:
??*
Taxis0*
Tindices0	*
Tparams0
t
bert/embeddings/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*!
valueB"   ?      
?
bert/embeddings/Reshape_1Reshapebert/embeddings/GatherV2bert/embeddings/Reshape_1/shape*
T0*
Tshape0*$
_output_shapes
:??
?
Hbert/embeddings/token_type_embeddings/Initializer/truncated_normal/shapeConst*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
valueB"      *
dtype0*
_output_shapes
:
?
Gbert/embeddings/token_type_embeddings/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
valueB
 *    
?
Ibert/embeddings/token_type_embeddings/Initializer/truncated_normal/stddevConst*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
Rbert/embeddings/token_type_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHbert/embeddings/token_type_embeddings/Initializer/truncated_normal/shape*
seed2 *
dtype0*
_output_shapes
:	?*

seed *
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings
?
Fbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mulMulRbert/embeddings/token_type_embeddings/Initializer/truncated_normal/TruncatedNormalIbert/embeddings/token_type_embeddings/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	?
?
Bbert/embeddings/token_type_embeddings/Initializer/truncated_normalAddFbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mulGbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	?
?
%bert/embeddings/token_type_embeddings
VariableV2*
	container *
shape:	?*
dtype0*
_output_shapes
:	?*
shared_name *8
_class.
,*loc:@bert/embeddings/token_type_embeddings
?
,bert/embeddings/token_type_embeddings/AssignAssign%bert/embeddings/token_type_embeddingsBbert/embeddings/token_type_embeddings/Initializer/truncated_normal*
use_locking(*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
validate_shape(*
_output_shapes
:	?
?
*bert/embeddings/token_type_embeddings/readIdentity%bert/embeddings/token_type_embeddings*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	?
r
bert/embeddings/Reshape_2/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
bert/embeddings/Reshape_2Reshapesegment_idsbert/embeddings/Reshape_2/shape*
T0	*
Tshape0*
_output_shapes	
:?
e
 bert/embeddings/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
f
!bert/embeddings/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
bert/embeddings/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
?
bert/embeddings/one_hotOneHotbert/embeddings/Reshape_2bert/embeddings/one_hot/depth bert/embeddings/one_hot/on_value!bert/embeddings/one_hot/off_value*
T0*
axis?????????*
TI0	*
_output_shapes
:	?
?
bert/embeddings/MatMulMatMulbert/embeddings/one_hot*bert/embeddings/token_type_embeddings/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
t
bert/embeddings/Reshape_3/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/embeddings/Reshape_3Reshapebert/embeddings/MatMulbert/embeddings/Reshape_3/shape*$
_output_shapes
:??*
T0*
Tshape0

bert/embeddings/addAddbert/embeddings/Reshape_1bert/embeddings/Reshape_3*
T0*$
_output_shapes
:??
f
#bert/embeddings/assert_less_equal/xConst*
value
B :?*
dtype0*
_output_shapes
: 
f
#bert/embeddings/assert_less_equal/yConst*
_output_shapes
: *
value
B :?*
dtype0
?
+bert/embeddings/assert_less_equal/LessEqual	LessEqual#bert/embeddings/assert_less_equal/x#bert/embeddings/assert_less_equal/y*
T0*
_output_shapes
: 
j
'bert/embeddings/assert_less_equal/ConstConst*
dtype0*
_output_shapes
: *
valueB 
?
%bert/embeddings/assert_less_equal/AllAll+bert/embeddings/assert_less_equal/LessEqual'bert/embeddings/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
o
.bert/embeddings/assert_less_equal/Assert/ConstConst*
dtype0*
_output_shapes
: *
valueB B 
?
0bert/embeddings/assert_less_equal/Assert/Const_1Const*h
value_B] BWCondition x <= y did not hold element-wise:x (bert/embeddings/assert_less_equal/x:0) = *
dtype0*
_output_shapes
: 
?
0bert/embeddings/assert_less_equal/Assert/Const_2Const*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = *
dtype0*
_output_shapes
: 
w
6bert/embeddings/assert_less_equal/Assert/Assert/data_0Const*
valueB B *
dtype0*
_output_shapes
: 
?
6bert/embeddings/assert_less_equal/Assert/Assert/data_1Const*h
value_B] BWCondition x <= y did not hold element-wise:x (bert/embeddings/assert_less_equal/x:0) = *
dtype0*
_output_shapes
: 
?
6bert/embeddings/assert_less_equal/Assert/Assert/data_3Const*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = *
dtype0*
_output_shapes
: 
?
/bert/embeddings/assert_less_equal/Assert/AssertAssert%bert/embeddings/assert_less_equal/All6bert/embeddings/assert_less_equal/Assert/Assert/data_06bert/embeddings/assert_less_equal/Assert/Assert/data_1#bert/embeddings/assert_less_equal/x6bert/embeddings/assert_less_equal/Assert/Assert/data_3#bert/embeddings/assert_less_equal/y*
T	
2*
	summarize
?
Fbert/embeddings/position_embeddings/Initializer/truncated_normal/shapeConst*6
_class,
*(loc:@bert/embeddings/position_embeddings*
valueB"      *
dtype0*
_output_shapes
:
?
Ebert/embeddings/position_embeddings/Initializer/truncated_normal/meanConst*6
_class,
*(loc:@bert/embeddings/position_embeddings*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Gbert/embeddings/position_embeddings/Initializer/truncated_normal/stddevConst*6
_class,
*(loc:@bert/embeddings/position_embeddings*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
Pbert/embeddings/position_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalFbert/embeddings/position_embeddings/Initializer/truncated_normal/shape*

seed *
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings*
seed2 *
dtype0* 
_output_shapes
:
??
?
Dbert/embeddings/position_embeddings/Initializer/truncated_normal/mulMulPbert/embeddings/position_embeddings/Initializer/truncated_normal/TruncatedNormalGbert/embeddings/position_embeddings/Initializer/truncated_normal/stddev*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
??
?
@bert/embeddings/position_embeddings/Initializer/truncated_normalAddDbert/embeddings/position_embeddings/Initializer/truncated_normal/mulEbert/embeddings/position_embeddings/Initializer/truncated_normal/mean*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
??
?
#bert/embeddings/position_embeddings
VariableV2*
shared_name *6
_class,
*(loc:@bert/embeddings/position_embeddings*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
*bert/embeddings/position_embeddings/AssignAssign#bert/embeddings/position_embeddings@bert/embeddings/position_embeddings/Initializer/truncated_normal*
use_locking(*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings*
validate_shape(* 
_output_shapes
:
??
?
(bert/embeddings/position_embeddings/readIdentity#bert/embeddings/position_embeddings*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
??
?
bert/embeddings/Slice/beginConst0^bert/embeddings/assert_less_equal/Assert/Assert*
_output_shapes
:*
valueB"        *
dtype0
?
bert/embeddings/Slice/sizeConst0^bert/embeddings/assert_less_equal/Assert/Assert*
valueB"?   ????*
dtype0*
_output_shapes
:
?
bert/embeddings/SliceSlice(bert/embeddings/position_embeddings/readbert/embeddings/Slice/beginbert/embeddings/Slice/size*
Index0*
T0* 
_output_shapes
:
??
?
bert/embeddings/Reshape_4/shapeConst0^bert/embeddings/assert_less_equal/Assert/Assert*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/embeddings/Reshape_4Reshapebert/embeddings/Slicebert/embeddings/Reshape_4/shape*$
_output_shapes
:??*
T0*
Tshape0
{
bert/embeddings/add_1Addbert/embeddings/addbert/embeddings/Reshape_4*
T0*$
_output_shapes
:??
?
0bert/embeddings/LayerNorm/beta/Initializer/zerosConst*
_output_shapes	
:?*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
valueB?*    *
dtype0
?
bert/embeddings/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
	container *
shape:?
?
%bert/embeddings/LayerNorm/beta/AssignAssignbert/embeddings/LayerNorm/beta0bert/embeddings/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta
?
#bert/embeddings/LayerNorm/beta/readIdentitybert/embeddings/LayerNorm/beta*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
_output_shapes	
:?*
T0
?
0bert/embeddings/LayerNorm/gamma/Initializer/onesConst*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
bert/embeddings/LayerNorm/gamma
VariableV2*
shared_name *2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
&bert/embeddings/LayerNorm/gamma/AssignAssignbert/embeddings/LayerNorm/gamma0bert/embeddings/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma
?
$bert/embeddings/LayerNorm/gamma/readIdentitybert/embeddings/LayerNorm/gamma*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
_output_shapes	
:?
?
8bert/embeddings/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
&bert/embeddings/LayerNorm/moments/meanMeanbert/embeddings/add_18bert/embeddings/LayerNorm/moments/mean/reduction_indices*#
_output_shapes
:?*
	keep_dims(*

Tidx0*
T0
?
.bert/embeddings/LayerNorm/moments/StopGradientStopGradient&bert/embeddings/LayerNorm/moments/mean*
T0*#
_output_shapes
:?
?
3bert/embeddings/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/embeddings/add_1.bert/embeddings/LayerNorm/moments/StopGradient*$
_output_shapes
:??*
T0
?
<bert/embeddings/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
?
*bert/embeddings/LayerNorm/moments/varianceMean3bert/embeddings/LayerNorm/moments/SquaredDifference<bert/embeddings/LayerNorm/moments/variance/reduction_indices*
T0*#
_output_shapes
:?*
	keep_dims(*

Tidx0
n
)bert/embeddings/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *̼?+*
dtype0
?
'bert/embeddings/LayerNorm/batchnorm/addAdd*bert/embeddings/LayerNorm/moments/variance)bert/embeddings/LayerNorm/batchnorm/add/y*#
_output_shapes
:?*
T0
?
)bert/embeddings/LayerNorm/batchnorm/RsqrtRsqrt'bert/embeddings/LayerNorm/batchnorm/add*#
_output_shapes
:?*
T0
?
'bert/embeddings/LayerNorm/batchnorm/mulMul)bert/embeddings/LayerNorm/batchnorm/Rsqrt$bert/embeddings/LayerNorm/gamma/read*$
_output_shapes
:??*
T0
?
)bert/embeddings/LayerNorm/batchnorm/mul_1Mulbert/embeddings/add_1'bert/embeddings/LayerNorm/batchnorm/mul*$
_output_shapes
:??*
T0
?
)bert/embeddings/LayerNorm/batchnorm/mul_2Mul&bert/embeddings/LayerNorm/moments/mean'bert/embeddings/LayerNorm/batchnorm/mul*
T0*$
_output_shapes
:??
?
'bert/embeddings/LayerNorm/batchnorm/subSub#bert/embeddings/LayerNorm/beta/read)bert/embeddings/LayerNorm/batchnorm/mul_2*$
_output_shapes
:??*
T0
?
)bert/embeddings/LayerNorm/batchnorm/add_1Add)bert/embeddings/LayerNorm/batchnorm/mul_1'bert/embeddings/LayerNorm/batchnorm/sub*
T0*$
_output_shapes
:??
o
bert/encoder/Reshape/shapeConst*!
valueB"      ?   *
dtype0*
_output_shapes
:
?
bert/encoder/ReshapeReshape
input_maskbert/encoder/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:?
|
bert/encoder/CastCastbert/encoder/Reshape*

SrcT0	*
Truncate( *#
_output_shapes
:?*

DstT0
p
bert/encoder/onesConst*#
_output_shapes
:?*"
valueB?*  ??*
dtype0
l
bert/encoder/mulMulbert/encoder/onesbert/encoder/Cast*
T0*$
_output_shapes
:??
m
bert/encoder/Reshape_1/shapeConst*
_output_shapes
:*
valueB"????   *
dtype0
?
bert/encoder/Reshape_1Reshape)bert/embeddings/LayerNorm/batchnorm/add_1bert/encoder/Reshape_1/shape*
T0*
Tshape0* 
_output_shapes
:
??
?
Sbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
valueB
 *    *
dtype0
?
Tbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
seed2 
?
Qbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_0/attention/self/query/kernel
VariableV2*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
7bert/encoder/layer_0/attention/self/query/kernel/AssignAssign0bert/encoder/layer_0/attention/self/query/kernelMbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_0/attention/self/query/kernel/readIdentity0bert/encoder/layer_0/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_0/attention/self/query/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
valueB?*    
?
.bert/encoder/layer_0/attention/self/query/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
	container *
shape:?
?
5bert/encoder/layer_0/attention/self/query/bias/AssignAssign.bert/encoder/layer_0/attention/self/query/bias@bert/encoder/layer_0/attention/self/query/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
validate_shape(
?
3bert/encoder/layer_0/attention/self/query/bias/readIdentity.bert/encoder/layer_0/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
_output_shapes	
:?
?
0bert/encoder/layer_0/attention/self/query/MatMulMatMulbert/encoder/Reshape_15bert/encoder/layer_0/attention/self/query/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
1bert/encoder/layer_0/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_0/attention/self/query/MatMul3bert/encoder/layer_0/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Qbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/shape*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
??
?
Obert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_0/attention/self/key/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
	container 
?
5bert/encoder/layer_0/attention/self/key/kernel/AssignAssign.bert/encoder/layer_0/attention/self/key/kernelKbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel
?
3bert/encoder/layer_0/attention/self/key/kernel/readIdentity.bert/encoder/layer_0/attention/self/key/kernel*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:
??*
T0
?
>bert/encoder/layer_0/attention/self/key/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
,bert/encoder/layer_0/attention/self/key/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
	container *
shape:?
?
3bert/encoder/layer_0/attention/self/key/bias/AssignAssign,bert/encoder/layer_0/attention/self/key/bias>bert/encoder/layer_0/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_0/attention/self/key/bias/readIdentity,bert/encoder/layer_0/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
_output_shapes	
:?
?
.bert/encoder/layer_0/attention/self/key/MatMulMatMulbert/encoder/Reshape_13bert/encoder/layer_0/attention/self/key/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
/bert/encoder/layer_0/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_0/attention/self/key/MatMul1bert/encoder/layer_0/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Sbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
valueB"      
?
Rbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
seed2 
?
Qbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel
?
0bert/encoder/layer_0/attention/self/value/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
	container *
shape:
??
?
7bert/encoder/layer_0/attention/self/value/kernel/AssignAssign0bert/encoder/layer_0/attention/self/value/kernelMbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
validate_shape(
?
5bert/encoder/layer_0/attention/self/value/kernel/readIdentity0bert/encoder/layer_0/attention/self/value/kernel* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel
?
@bert/encoder/layer_0/attention/self/value/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_0/attention/self/value/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
	container *
shape:?
?
5bert/encoder/layer_0/attention/self/value/bias/AssignAssign.bert/encoder/layer_0/attention/self/value/bias@bert/encoder/layer_0/attention/self/value/bias/Initializer/zeros*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
3bert/encoder/layer_0/attention/self/value/bias/readIdentity.bert/encoder/layer_0/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
_output_shapes	
:?
?
0bert/encoder/layer_0/attention/self/value/MatMulMatMulbert/encoder/Reshape_15bert/encoder/layer_0/attention/self/value/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
1bert/encoder/layer_0/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_0/attention/self/value/MatMul3bert/encoder/layer_0/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
1bert/encoder/layer_0/attention/self/Reshape/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
+bert/encoder/layer_0/attention/self/ReshapeReshape1bert/encoder/layer_0/attention/self/query/BiasAdd1bert/encoder/layer_0/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
2bert/encoder/layer_0/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_0/attention/self/transpose	Transpose+bert/encoder/layer_0/attention/self/Reshape2bert/encoder/layer_0/attention/self/transpose/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
3bert/encoder/layer_0/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_0/attention/self/Reshape_1Reshape/bert/encoder/layer_0/attention/self/key/BiasAdd3bert/encoder/layer_0/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_0/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_0/attention/self/transpose_1	Transpose-bert/encoder/layer_0/attention/self/Reshape_14bert/encoder/layer_0/attention/self/transpose_1/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
*bert/encoder/layer_0/attention/self/MatMulBatchMatMul-bert/encoder/layer_0/attention/self/transpose/bert/encoder/layer_0/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:??
n
)bert/encoder/layer_0/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_0/attention/self/MulMul*bert/encoder/layer_0/attention/self/MatMul)bert/encoder/layer_0/attention/self/Mul/y*
T0*(
_output_shapes
:??
|
2bert/encoder/layer_0/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
.bert/encoder/layer_0/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_0/attention/self/ExpandDims/dim*(
_output_shapes
:??*

Tdim0*
T0
n
)bert/encoder/layer_0/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_0/attention/self/subSub)bert/encoder/layer_0/attention/self/sub/x.bert/encoder/layer_0/attention/self/ExpandDims*
T0*(
_output_shapes
:??
p
+bert/encoder/layer_0/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
)bert/encoder/layer_0/attention/self/mul_1Mul'bert/encoder/layer_0/attention/self/sub+bert/encoder/layer_0/attention/self/mul_1/y*
T0*(
_output_shapes
:??
?
'bert/encoder/layer_0/attention/self/addAdd'bert/encoder/layer_0/attention/self/Mul)bert/encoder/layer_0/attention/self/mul_1*
T0*(
_output_shapes
:??
?
+bert/encoder/layer_0/attention/self/SoftmaxSoftmax'bert/encoder/layer_0/attention/self/add*
T0*(
_output_shapes
:??
?
3bert/encoder/layer_0/attention/self/Reshape_2/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_0/attention/self/Reshape_2Reshape1bert/encoder/layer_0/attention/self/value/BiasAdd3bert/encoder/layer_0/attention/self/Reshape_2/shape*
Tshape0*'
_output_shapes
:?@*
T0
?
4bert/encoder/layer_0/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_0/attention/self/transpose_2	Transpose-bert/encoder/layer_0/attention/self/Reshape_24bert/encoder/layer_0/attention/self/transpose_2/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
,bert/encoder/layer_0/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_0/attention/self/Softmax/bert/encoder/layer_0/attention/self/transpose_2*
adj_x( *
adj_y( *
T0*'
_output_shapes
:?@
?
4bert/encoder/layer_0/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_0/attention/self/transpose_3	Transpose,bert/encoder/layer_0/attention/self/MatMul_14bert/encoder/layer_0/attention/self/transpose_3/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_0/attention/self/Reshape_3/shapeConst*
_output_shapes
:*
valueB"?      *
dtype0
?
-bert/encoder/layer_0/attention/self/Reshape_3Reshape/bert/encoder/layer_0/attention/self/transpose_33bert/encoder/layer_0/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:
??
?
Ubert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Tbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
valueB
 *    *
dtype0
?
Vbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
_bert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel
?
Sbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel
?
2bert/encoder/layer_0/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
	container *
shape:
??
?
9bert/encoder/layer_0/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_0/attention/output/dense/kernelObert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
7bert/encoder/layer_0/attention/output/dense/kernel/readIdentity2bert/encoder/layer_0/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel* 
_output_shapes
:
??
?
Bbert/encoder/layer_0/attention/output/dense/bias/Initializer/zerosConst*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
0bert/encoder/layer_0/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
	container *
shape:?
?
7bert/encoder/layer_0/attention/output/dense/bias/AssignAssign0bert/encoder/layer_0/attention/output/dense/biasBbert/encoder/layer_0/attention/output/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias
?
5bert/encoder/layer_0/attention/output/dense/bias/readIdentity0bert/encoder/layer_0/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
_output_shapes	
:?
?
2bert/encoder/layer_0/attention/output/dense/MatMulMatMul-bert/encoder/layer_0/attention/self/Reshape_37bert/encoder/layer_0/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
3bert/encoder/layer_0/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_0/attention/output/dense/MatMul5bert/encoder/layer_0/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
)bert/encoder/layer_0/attention/output/addAdd3bert/encoder/layer_0/attention/output/dense/BiasAddbert/encoder/Reshape_1*
T0* 
_output_shapes
:
??
?
Fbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zerosConst*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
4bert/encoder/layer_0/attention/output/LayerNorm/beta
VariableV2*
shared_name *G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
;bert/encoder/layer_0/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_0/attention/output/LayerNorm/betaFbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
validate_shape(
?
9bert/encoder/layer_0/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_0/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
_output_shapes	
:?
?
Fbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_0/attention/output/LayerNorm/gamma
VariableV2*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name 
?
<bert/encoder/layer_0/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_0/attention/output/LayerNorm/gammaFbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
:bert/encoder/layer_0/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_0/attention/output/LayerNorm/gamma*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
_output_shapes	
:?*
T0
?
Nbert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
<bert/encoder/layer_0/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_0/attention/output/addNbert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
Dbert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_0/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
Ibert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_0/attention/output/addDbert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Rbert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
@bert/encoder/layer_0/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *̼?+*
dtype0
?
=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_0/attention/output/LayerNorm/moments/variance?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	?*
T0
?
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_0/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_0/attention/output/add=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_0/attention/output/LayerNorm/moments/mean=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_0/attention/output/LayerNorm/beta/read?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Qbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
valueB"      
?
Pbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
seed2 *
dtype0
?
Obert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_0/intermediate/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
	container *
shape:
??
?
5bert/encoder/layer_0/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_0/intermediate/dense/kernelKbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
3bert/encoder/layer_0/intermediate/dense/kernel/readIdentity.bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel
?
Nbert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
valueB:?*
dtype0*
_output_shapes
:
?
Dbert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros/ConstConst*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>bert/encoder/layer_0/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros/Const*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*

index_type0
?
,bert/encoder/layer_0/intermediate/dense/bias
VariableV2*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
	container *
shape:?*
dtype0
?
3bert/encoder/layer_0/intermediate/dense/bias/AssignAssign,bert/encoder/layer_0/intermediate/dense/bias>bert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_0/intermediate/dense/bias/readIdentity,bert/encoder/layer_0/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
_output_shapes	
:?
?
.bert/encoder/layer_0/intermediate/dense/MatMulMatMul?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_0/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_0/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_0/intermediate/dense/MatMul1bert/encoder/layer_0/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
r
-bert/encoder/layer_0/intermediate/dense/Pow/yConst*
_output_shapes
: *
valueB
 *  @@*
dtype0
?
+bert/encoder/layer_0/intermediate/dense/PowPow/bert/encoder/layer_0/intermediate/dense/BiasAdd-bert/encoder/layer_0/intermediate/dense/Pow/y*
T0* 
_output_shapes
:
??
r
-bert/encoder/layer_0/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_0/intermediate/dense/mulMul-bert/encoder/layer_0/intermediate/dense/mul/x+bert/encoder/layer_0/intermediate/dense/Pow*
T0* 
_output_shapes
:
??
?
+bert/encoder/layer_0/intermediate/dense/addAdd/bert/encoder/layer_0/intermediate/dense/BiasAdd+bert/encoder/layer_0/intermediate/dense/mul*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_0/intermediate/dense/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 **BL?
?
-bert/encoder/layer_0/intermediate/dense/mul_1Mul/bert/encoder/layer_0/intermediate/dense/mul_1/x+bert/encoder/layer_0/intermediate/dense/add*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_0/intermediate/dense/TanhTanh-bert/encoder/layer_0/intermediate/dense/mul_1*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_0/intermediate/dense/add_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
-bert/encoder/layer_0/intermediate/dense/add_1Add/bert/encoder/layer_0/intermediate/dense/add_1/x,bert/encoder/layer_0/intermediate/dense/Tanh*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_0/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_0/intermediate/dense/mul_2Mul/bert/encoder/layer_0/intermediate/dense/mul_2/x-bert/encoder/layer_0/intermediate/dense/add_1*
T0* 
_output_shapes
:
??
?
-bert/encoder/layer_0/intermediate/dense/mul_3Mul/bert/encoder/layer_0/intermediate/dense/BiasAdd-bert/encoder/layer_0/intermediate/dense/mul_2*
T0* 
_output_shapes
:
??
?
Kbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
valueB"      
?
Jbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
valueB
 *    *
dtype0
?
Lbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
valueB
 *
ף<*
dtype0
?
Ubert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/shape*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Ibert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:
??
?
Ebert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel
?
(bert/encoder/layer_0/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
	container *
shape:
??
?
/bert/encoder/layer_0/output/dense/kernel/AssignAssign(bert/encoder/layer_0/output/dense/kernelEbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
-bert/encoder/layer_0/output/dense/kernel/readIdentity(bert/encoder/layer_0/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:
??
?
8bert/encoder/layer_0/output/dense/bias/Initializer/zerosConst*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
&bert/encoder/layer_0/output/dense/bias
VariableV2*
shared_name *9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
-bert/encoder/layer_0/output/dense/bias/AssignAssign&bert/encoder/layer_0/output/dense/bias8bert/encoder/layer_0/output/dense/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
validate_shape(
?
+bert/encoder/layer_0/output/dense/bias/readIdentity&bert/encoder/layer_0/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
_output_shapes	
:?
?
(bert/encoder/layer_0/output/dense/MatMulMatMul-bert/encoder/layer_0/intermediate/dense/mul_3-bert/encoder/layer_0/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
)bert/encoder/layer_0/output/dense/BiasAddBiasAdd(bert/encoder/layer_0/output/dense/MatMul+bert/encoder/layer_0/output/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
bert/encoder/layer_0/output/addAdd)bert/encoder/layer_0/output/dense/BiasAdd?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
<bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
*bert/encoder/layer_0/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
	container *
shape:?
?
1bert/encoder/layer_0/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_0/output/LayerNorm/beta<bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
/bert/encoder/layer_0/output/LayerNorm/beta/readIdentity*bert/encoder/layer_0/output/LayerNorm/beta*
_output_shapes	
:?*
T0*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta
?
<bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
+bert/encoder/layer_0/output/LayerNorm/gamma
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
	container 
?
2bert/encoder/layer_0/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_0/output/LayerNorm/gamma<bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
0bert/encoder/layer_0/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_0/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
_output_shapes	
:?
?
Dbert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
2bert/encoder/layer_0/output/LayerNorm/moments/meanMeanbert/encoder/layer_0/output/addDbert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
:bert/encoder/layer_0/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_0/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_0/output/add:bert/encoder/layer_0/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
??*
T0
?
Hbert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
?
6bert/encoder/layer_0/output/LayerNorm/moments/varianceMean?bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
z
5bert/encoder/layer_0/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
3bert/encoder/layer_0/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_0/output/LayerNorm/moments/variance5bert/encoder/layer_0/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
5bert/encoder/layer_0/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_0/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
3bert/encoder/layer_0/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_0/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_0/output/add3bert/encoder/layer_0/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_0/output/LayerNorm/moments/mean3bert/encoder/layer_0/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
3bert/encoder/layer_0/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_0/output/LayerNorm/beta/read5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_0/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Sbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
seed2 
?
Qbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_1/attention/self/query/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
	container 
?
7bert/encoder/layer_1/attention/self/query/kernel/AssignAssign0bert/encoder/layer_1/attention/self/query/kernelMbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
validate_shape(
?
5bert/encoder/layer_1/attention/self/query/kernel/readIdentity0bert/encoder/layer_1/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_1/attention/self/query/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_1/attention/self/query/bias
VariableV2*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
	container *
shape:?*
dtype0
?
5bert/encoder/layer_1/attention/self/query/bias/AssignAssign.bert/encoder/layer_1/attention/self/query/bias@bert/encoder/layer_1/attention/self/query/bias/Initializer/zeros*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
3bert/encoder/layer_1/attention/self/query/bias/readIdentity.bert/encoder/layer_1/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
_output_shapes	
:?
?
0bert/encoder/layer_1/attention/self/query/MatMulMatMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_15bert/encoder/layer_1/attention/self/query/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
1bert/encoder/layer_1/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_1/attention/self/query/MatMul3bert/encoder/layer_1/attention/self/query/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
Qbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
valueB"      *
dtype0
?
Pbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
valueB
 *    
?
Rbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
seed2 
?
Obert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel
?
Kbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_1/attention/self/key/kernel
VariableV2* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
	container *
shape:
??*
dtype0
?
5bert/encoder/layer_1/attention/self/key/kernel/AssignAssign.bert/encoder/layer_1/attention/self/key/kernelKbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
3bert/encoder/layer_1/attention/self/key/kernel/readIdentity.bert/encoder/layer_1/attention/self/key/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel* 
_output_shapes
:
??
?
>bert/encoder/layer_1/attention/self/key/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
valueB?*    
?
,bert/encoder/layer_1/attention/self/key/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
	container *
shape:?
?
3bert/encoder/layer_1/attention/self/key/bias/AssignAssign,bert/encoder/layer_1/attention/self/key/bias>bert/encoder/layer_1/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_1/attention/self/key/bias/readIdentity,bert/encoder/layer_1/attention/self/key/bias*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias
?
.bert/encoder/layer_1/attention/self/key/MatMulMatMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_13bert/encoder/layer_1/attention/self/key/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
/bert/encoder/layer_1/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_1/attention/self/key/MatMul1bert/encoder/layer_1/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Sbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
valueB
 *    
?
Tbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/shape*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0
?
Qbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel
?
Mbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_1/attention/self/value/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
	container *
shape:
??
?
7bert/encoder/layer_1/attention/self/value/kernel/AssignAssign0bert/encoder/layer_1/attention/self/value/kernelMbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
5bert/encoder/layer_1/attention/self/value/kernel/readIdentity0bert/encoder/layer_1/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_1/attention/self/value/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
valueB?*    
?
.bert/encoder/layer_1/attention/self/value/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_1/attention/self/value/bias/AssignAssign.bert/encoder/layer_1/attention/self/value/bias@bert/encoder/layer_1/attention/self/value/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
3bert/encoder/layer_1/attention/self/value/bias/readIdentity.bert/encoder/layer_1/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
_output_shapes	
:?
?
0bert/encoder/layer_1/attention/self/value/MatMulMatMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_15bert/encoder/layer_1/attention/self/value/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
1bert/encoder/layer_1/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_1/attention/self/value/MatMul3bert/encoder/layer_1/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
1bert/encoder/layer_1/attention/self/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?      @   
?
+bert/encoder/layer_1/attention/self/ReshapeReshape1bert/encoder/layer_1/attention/self/query/BiasAdd1bert/encoder/layer_1/attention/self/Reshape/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
2bert/encoder/layer_1/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_1/attention/self/transpose	Transpose+bert/encoder/layer_1/attention/self/Reshape2bert/encoder/layer_1/attention/self/transpose/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
3bert/encoder/layer_1/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_1/attention/self/Reshape_1Reshape/bert/encoder/layer_1/attention/self/key/BiasAdd3bert/encoder/layer_1/attention/self/Reshape_1/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
4bert/encoder/layer_1/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_1/attention/self/transpose_1	Transpose-bert/encoder/layer_1/attention/self/Reshape_14bert/encoder/layer_1/attention/self/transpose_1/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
*bert/encoder/layer_1/attention/self/MatMulBatchMatMul-bert/encoder/layer_1/attention/self/transpose/bert/encoder/layer_1/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:??
n
)bert/encoder/layer_1/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_1/attention/self/MulMul*bert/encoder/layer_1/attention/self/MatMul)bert/encoder/layer_1/attention/self/Mul/y*
T0*(
_output_shapes
:??
|
2bert/encoder/layer_1/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
.bert/encoder/layer_1/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_1/attention/self/ExpandDims/dim*
T0*(
_output_shapes
:??*

Tdim0
n
)bert/encoder/layer_1/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_1/attention/self/subSub)bert/encoder/layer_1/attention/self/sub/x.bert/encoder/layer_1/attention/self/ExpandDims*(
_output_shapes
:??*
T0
p
+bert/encoder/layer_1/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
)bert/encoder/layer_1/attention/self/mul_1Mul'bert/encoder/layer_1/attention/self/sub+bert/encoder/layer_1/attention/self/mul_1/y*
T0*(
_output_shapes
:??
?
'bert/encoder/layer_1/attention/self/addAdd'bert/encoder/layer_1/attention/self/Mul)bert/encoder/layer_1/attention/self/mul_1*
T0*(
_output_shapes
:??
?
+bert/encoder/layer_1/attention/self/SoftmaxSoftmax'bert/encoder/layer_1/attention/self/add*
T0*(
_output_shapes
:??
?
3bert/encoder/layer_1/attention/self/Reshape_2/shapeConst*
_output_shapes
:*%
valueB"   ?      @   *
dtype0
?
-bert/encoder/layer_1/attention/self/Reshape_2Reshape1bert/encoder/layer_1/attention/self/value/BiasAdd3bert/encoder/layer_1/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_1/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_1/attention/self/transpose_2	Transpose-bert/encoder/layer_1/attention/self/Reshape_24bert/encoder/layer_1/attention/self/transpose_2/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
,bert/encoder/layer_1/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_1/attention/self/Softmax/bert/encoder/layer_1/attention/self/transpose_2*'
_output_shapes
:?@*
adj_x( *
adj_y( *
T0
?
4bert/encoder/layer_1/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_1/attention/self/transpose_3	Transpose,bert/encoder/layer_1/attention/self/MatMul_14bert/encoder/layer_1/attention/self/transpose_3/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_1/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_1/attention/self/Reshape_3Reshape/bert/encoder/layer_1/attention/self/transpose_33bert/encoder/layer_1/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:
??
?
Ubert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Tbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0
?
_bert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/shape*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??
?
Sbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:
??
?
2bert/encoder/layer_1/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
	container *
shape:
??
?
9bert/encoder/layer_1/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_1/attention/output/dense/kernelObert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
7bert/encoder/layer_1/attention/output/dense/kernel/readIdentity2bert/encoder/layer_1/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:
??
?
Bbert/encoder/layer_1/attention/output/dense/bias/Initializer/zerosConst*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
0bert/encoder/layer_1/attention/output/dense/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
	container 
?
7bert/encoder/layer_1/attention/output/dense/bias/AssignAssign0bert/encoder/layer_1/attention/output/dense/biasBbert/encoder/layer_1/attention/output/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias
?
5bert/encoder/layer_1/attention/output/dense/bias/readIdentity0bert/encoder/layer_1/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
_output_shapes	
:?
?
2bert/encoder/layer_1/attention/output/dense/MatMulMatMul-bert/encoder/layer_1/attention/self/Reshape_37bert/encoder/layer_1/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
3bert/encoder/layer_1/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_1/attention/output/dense/MatMul5bert/encoder/layer_1/attention/output/dense/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
?
)bert/encoder/layer_1/attention/output/addAdd3bert/encoder/layer_1/attention/output/dense/BiasAdd5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
Fbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
valueB?*    
?
4bert/encoder/layer_1/attention/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
	container *
shape:?
?
;bert/encoder/layer_1/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_1/attention/output/LayerNorm/betaFbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
9bert/encoder/layer_1/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_1/attention/output/LayerNorm/beta*
_output_shapes	
:?*
T0*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta
?
Fbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_1/attention/output/LayerNorm/gamma
VariableV2*
shared_name *H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
<bert/encoder/layer_1/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_1/attention/output/LayerNorm/gammaFbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
:bert/encoder/layer_1/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_1/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
_output_shapes	
:?
?
Nbert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
<bert/encoder/layer_1/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_1/attention/output/addNbert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
Dbert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_1/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
Ibert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_1/attention/output/addDbert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Rbert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
@bert/encoder/layer_1/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼?+
?
=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_1/attention/output/LayerNorm/moments/variance?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	?*
T0
?
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add*
_output_shapes
:	?*
T0
?
=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_1/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_1/attention/output/add=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_1/attention/output/LayerNorm/moments/mean=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_1/attention/output/LayerNorm/beta/read?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Qbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
seed2 
?
Obert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_1/intermediate/dense/kernel
VariableV2*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel
?
5bert/encoder/layer_1/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_1/intermediate/dense/kernelKbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
3bert/encoder/layer_1/intermediate/dense/kernel/readIdentity.bert/encoder/layer_1/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
valueB:?*
dtype0*
_output_shapes
:
?
Dbert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros/ConstConst*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>bert/encoder/layer_1/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros/Const*
T0*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*

index_type0*
_output_shapes	
:?
?
,bert/encoder/layer_1/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
	container *
shape:?
?
3bert/encoder/layer_1/intermediate/dense/bias/AssignAssign,bert/encoder/layer_1/intermediate/dense/bias>bert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
1bert/encoder/layer_1/intermediate/dense/bias/readIdentity,bert/encoder/layer_1/intermediate/dense/bias*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias
?
.bert/encoder/layer_1/intermediate/dense/MatMulMatMul?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_1/intermediate/dense/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
/bert/encoder/layer_1/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_1/intermediate/dense/MatMul1bert/encoder/layer_1/intermediate/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
r
-bert/encoder/layer_1/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_1/intermediate/dense/PowPow/bert/encoder/layer_1/intermediate/dense/BiasAdd-bert/encoder/layer_1/intermediate/dense/Pow/y*
T0* 
_output_shapes
:
??
r
-bert/encoder/layer_1/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_1/intermediate/dense/mulMul-bert/encoder/layer_1/intermediate/dense/mul/x+bert/encoder/layer_1/intermediate/dense/Pow*
T0* 
_output_shapes
:
??
?
+bert/encoder/layer_1/intermediate/dense/addAdd/bert/encoder/layer_1/intermediate/dense/BiasAdd+bert/encoder/layer_1/intermediate/dense/mul*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_1/intermediate/dense/mul_1/xConst*
_output_shapes
: *
valueB
 **BL?*
dtype0
?
-bert/encoder/layer_1/intermediate/dense/mul_1Mul/bert/encoder/layer_1/intermediate/dense/mul_1/x+bert/encoder/layer_1/intermediate/dense/add*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_1/intermediate/dense/TanhTanh-bert/encoder/layer_1/intermediate/dense/mul_1* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_1/intermediate/dense/add_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_1/intermediate/dense/add_1Add/bert/encoder/layer_1/intermediate/dense/add_1/x,bert/encoder/layer_1/intermediate/dense/Tanh*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_1/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_1/intermediate/dense/mul_2Mul/bert/encoder/layer_1/intermediate/dense/mul_2/x-bert/encoder/layer_1/intermediate/dense/add_1*
T0* 
_output_shapes
:
??
?
-bert/encoder/layer_1/intermediate/dense/mul_3Mul/bert/encoder/layer_1/intermediate/dense/BiasAdd-bert/encoder/layer_1/intermediate/dense/mul_2* 
_output_shapes
:
??*
T0
?
Kbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Jbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Lbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/stddevConst*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
Ubert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
seed2 
?
Ibert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:
??
?
Ebert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:
??
?
(bert/encoder/layer_1/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
	container *
shape:
??
?
/bert/encoder/layer_1/output/dense/kernel/AssignAssign(bert/encoder/layer_1/output/dense/kernelEbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel
?
-bert/encoder/layer_1/output/dense/kernel/readIdentity(bert/encoder/layer_1/output/dense/kernel*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:
??*
T0
?
8bert/encoder/layer_1/output/dense/bias/Initializer/zerosConst*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
&bert/encoder/layer_1/output/dense/bias
VariableV2*
shared_name *9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
-bert/encoder/layer_1/output/dense/bias/AssignAssign&bert/encoder/layer_1/output/dense/bias8bert/encoder/layer_1/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
+bert/encoder/layer_1/output/dense/bias/readIdentity&bert/encoder/layer_1/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
_output_shapes	
:?
?
(bert/encoder/layer_1/output/dense/MatMulMatMul-bert/encoder/layer_1/intermediate/dense/mul_3-bert/encoder/layer_1/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
)bert/encoder/layer_1/output/dense/BiasAddBiasAdd(bert/encoder/layer_1/output/dense/MatMul+bert/encoder/layer_1/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
bert/encoder/layer_1/output/addAdd)bert/encoder/layer_1/output/dense/BiasAdd?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
<bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
*bert/encoder/layer_1/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
	container *
shape:?
?
1bert/encoder/layer_1/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_1/output/LayerNorm/beta<bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
/bert/encoder/layer_1/output/LayerNorm/beta/readIdentity*bert/encoder/layer_1/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
_output_shapes	
:?
?
<bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
+bert/encoder/layer_1/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
	container *
shape:?
?
2bert/encoder/layer_1/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_1/output/LayerNorm/gamma<bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
0bert/encoder/layer_1/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_1/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
_output_shapes	
:?
?
Dbert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
2bert/encoder/layer_1/output/LayerNorm/moments/meanMeanbert/encoder/layer_1/output/addDbert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
:bert/encoder/layer_1/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_1/output/LayerNorm/moments/mean*
_output_shapes
:	?*
T0
?
?bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_1/output/add:bert/encoder/layer_1/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Hbert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
6bert/encoder/layer_1/output/LayerNorm/moments/varianceMean?bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
z
5bert/encoder/layer_1/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
3bert/encoder/layer_1/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_1/output/LayerNorm/moments/variance5bert/encoder/layer_1/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
5bert/encoder/layer_1/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_1/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
3bert/encoder/layer_1/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_1/output/LayerNorm/gamma/read* 
_output_shapes
:
??*
T0
?
5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_1/output/add3bert/encoder/layer_1/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_1/output/LayerNorm/moments/mean3bert/encoder/layer_1/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
3bert/encoder/layer_1/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_1/output/LayerNorm/beta/read5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_1/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
??*
T0
?
Sbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Qbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/stddev*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel* 
_output_shapes
:
??*
T0
?
Mbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_2/attention/self/query/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
7bert/encoder/layer_2/attention/self/query/kernel/AssignAssign0bert/encoder/layer_2/attention/self/query/kernelMbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
5bert/encoder/layer_2/attention/self/query/kernel/readIdentity0bert/encoder/layer_2/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_2/attention/self/query/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_2/attention/self/query/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
	container 
?
5bert/encoder/layer_2/attention/self/query/bias/AssignAssign.bert/encoder/layer_2/attention/self/query/bias@bert/encoder/layer_2/attention/self/query/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
validate_shape(
?
3bert/encoder/layer_2/attention/self/query/bias/readIdentity.bert/encoder/layer_2/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
_output_shapes	
:?
?
0bert/encoder/layer_2/attention/self/query/MatMulMatMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_15bert/encoder/layer_2/attention/self/query/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
1bert/encoder/layer_2/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_2/attention/self/query/MatMul3bert/encoder/layer_2/attention/self/query/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
Qbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
valueB
 *    
?
Rbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
seed2 
?
Obert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel
?
Kbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_2/attention/self/key/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
	container *
shape:
??
?
5bert/encoder/layer_2/attention/self/key/kernel/AssignAssign.bert/encoder/layer_2/attention/self/key/kernelKbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
validate_shape(
?
3bert/encoder/layer_2/attention/self/key/kernel/readIdentity.bert/encoder/layer_2/attention/self/key/kernel* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel
?
>bert/encoder/layer_2/attention/self/key/bias/Initializer/zerosConst*
_output_shapes	
:?*?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
valueB?*    *
dtype0
?
,bert/encoder/layer_2/attention/self/key/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
	container 
?
3bert/encoder/layer_2/attention/self/key/bias/AssignAssign,bert/encoder/layer_2/attention/self/key/bias>bert/encoder/layer_2/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_2/attention/self/key/bias/readIdentity,bert/encoder/layer_2/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
_output_shapes	
:?
?
.bert/encoder/layer_2/attention/self/key/MatMulMatMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_13bert/encoder/layer_2/attention/self/key/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_2/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_2/attention/self/key/MatMul1bert/encoder/layer_2/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Sbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
valueB
 *
ף<
?
]bert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
seed2 
?
Qbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_2/attention/self/value/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
	container *
shape:
??
?
7bert/encoder/layer_2/attention/self/value/kernel/AssignAssign0bert/encoder/layer_2/attention/self/value/kernelMbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_2/attention/self/value/kernel/readIdentity0bert/encoder/layer_2/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_2/attention/self/value/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
valueB?*    
?
.bert/encoder/layer_2/attention/self/value/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
	container *
shape:?
?
5bert/encoder/layer_2/attention/self/value/bias/AssignAssign.bert/encoder/layer_2/attention/self/value/bias@bert/encoder/layer_2/attention/self/value/bias/Initializer/zeros*A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
3bert/encoder/layer_2/attention/self/value/bias/readIdentity.bert/encoder/layer_2/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
_output_shapes	
:?
?
0bert/encoder/layer_2/attention/self/value/MatMulMatMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_15bert/encoder/layer_2/attention/self/value/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
1bert/encoder/layer_2/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_2/attention/self/value/MatMul3bert/encoder/layer_2/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
1bert/encoder/layer_2/attention/self/Reshape/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
+bert/encoder/layer_2/attention/self/ReshapeReshape1bert/encoder/layer_2/attention/self/query/BiasAdd1bert/encoder/layer_2/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
2bert/encoder/layer_2/attention/self/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
?
-bert/encoder/layer_2/attention/self/transpose	Transpose+bert/encoder/layer_2/attention/self/Reshape2bert/encoder/layer_2/attention/self/transpose/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
3bert/encoder/layer_2/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_2/attention/self/Reshape_1Reshape/bert/encoder/layer_2/attention/self/key/BiasAdd3bert/encoder/layer_2/attention/self/Reshape_1/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
4bert/encoder/layer_2/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_2/attention/self/transpose_1	Transpose-bert/encoder/layer_2/attention/self/Reshape_14bert/encoder/layer_2/attention/self/transpose_1/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
*bert/encoder/layer_2/attention/self/MatMulBatchMatMul-bert/encoder/layer_2/attention/self/transpose/bert/encoder/layer_2/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:??
n
)bert/encoder/layer_2/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_2/attention/self/MulMul*bert/encoder/layer_2/attention/self/MatMul)bert/encoder/layer_2/attention/self/Mul/y*(
_output_shapes
:??*
T0
|
2bert/encoder/layer_2/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
.bert/encoder/layer_2/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_2/attention/self/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:??
n
)bert/encoder/layer_2/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_2/attention/self/subSub)bert/encoder/layer_2/attention/self/sub/x.bert/encoder/layer_2/attention/self/ExpandDims*
T0*(
_output_shapes
:??
p
+bert/encoder/layer_2/attention/self/mul_1/yConst*
dtype0*
_output_shapes
: *
valueB
 * @?
?
)bert/encoder/layer_2/attention/self/mul_1Mul'bert/encoder/layer_2/attention/self/sub+bert/encoder/layer_2/attention/self/mul_1/y*(
_output_shapes
:??*
T0
?
'bert/encoder/layer_2/attention/self/addAdd'bert/encoder/layer_2/attention/self/Mul)bert/encoder/layer_2/attention/self/mul_1*
T0*(
_output_shapes
:??
?
+bert/encoder/layer_2/attention/self/SoftmaxSoftmax'bert/encoder/layer_2/attention/self/add*
T0*(
_output_shapes
:??
?
3bert/encoder/layer_2/attention/self/Reshape_2/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_2/attention/self/Reshape_2Reshape1bert/encoder/layer_2/attention/self/value/BiasAdd3bert/encoder/layer_2/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_2/attention/self/transpose_2/permConst*
_output_shapes
:*%
valueB"             *
dtype0
?
/bert/encoder/layer_2/attention/self/transpose_2	Transpose-bert/encoder/layer_2/attention/self/Reshape_24bert/encoder/layer_2/attention/self/transpose_2/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
,bert/encoder/layer_2/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_2/attention/self/Softmax/bert/encoder/layer_2/attention/self/transpose_2*'
_output_shapes
:?@*
adj_x( *
adj_y( *
T0
?
4bert/encoder/layer_2/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_2/attention/self/transpose_3	Transpose,bert/encoder/layer_2/attention/self/MatMul_14bert/encoder/layer_2/attention/self/transpose_3/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
3bert/encoder/layer_2/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_2/attention/self/Reshape_3Reshape/bert/encoder/layer_2/attention/self/transpose_33bert/encoder/layer_2/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:
??
?
Ubert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Tbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
valueB
 *    
?
Vbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0
?
_bert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/shape*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0
?
Sbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel* 
_output_shapes
:
??
?
2bert/encoder/layer_2/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
	container *
shape:
??
?
9bert/encoder/layer_2/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_2/attention/output/dense/kernelObert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
7bert/encoder/layer_2/attention/output/dense/kernel/readIdentity2bert/encoder/layer_2/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel* 
_output_shapes
:
??
?
Bbert/encoder/layer_2/attention/output/dense/bias/Initializer/zerosConst*
_output_shapes	
:?*C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
valueB?*    *
dtype0
?
0bert/encoder/layer_2/attention/output/dense/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
	container 
?
7bert/encoder/layer_2/attention/output/dense/bias/AssignAssign0bert/encoder/layer_2/attention/output/dense/biasBbert/encoder/layer_2/attention/output/dense/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
5bert/encoder/layer_2/attention/output/dense/bias/readIdentity0bert/encoder/layer_2/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
_output_shapes	
:?
?
2bert/encoder/layer_2/attention/output/dense/MatMulMatMul-bert/encoder/layer_2/attention/self/Reshape_37bert/encoder/layer_2/attention/output/dense/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
3bert/encoder/layer_2/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_2/attention/output/dense/MatMul5bert/encoder/layer_2/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
)bert/encoder/layer_2/attention/output/addAdd3bert/encoder/layer_2/attention/output/dense/BiasAdd5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
Fbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zerosConst*G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
4bert/encoder/layer_2/attention/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
	container *
shape:?
?
;bert/encoder/layer_2/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_2/attention/output/LayerNorm/betaFbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
9bert/encoder/layer_2/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_2/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
_output_shapes	
:?
?
Fbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_2/attention/output/LayerNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma
?
<bert/encoder/layer_2/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_2/attention/output/LayerNorm/gammaFbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/ones*
T0*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
:bert/encoder/layer_2/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_2/attention/output/LayerNorm/gamma*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
_output_shapes	
:?*
T0
?
Nbert/encoder/layer_2/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
?
<bert/encoder/layer_2/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_2/attention/output/addNbert/encoder/layer_2/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
Dbert/encoder/layer_2/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_2/attention/output/LayerNorm/moments/mean*
_output_shapes
:	?*
T0
?
Ibert/encoder/layer_2/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_2/attention/output/addDbert/encoder/layer_2/attention/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
??*
T0
?
Rbert/encoder/layer_2/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
@bert/encoder/layer_2/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_2/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_2/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_2/attention/output/LayerNorm/moments/variance?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_2/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_2/attention/output/add=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_2/attention/output/LayerNorm/moments/mean=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_2/attention/output/LayerNorm/beta/read?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
??*
T0
?
Qbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/shape*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??
?
Obert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_2/intermediate/dense/kernel
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
5bert/encoder/layer_2/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_2/intermediate/dense/kernelKbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
3bert/encoder/layer_2/intermediate/dense/kernel/readIdentity.bert/encoder/layer_2/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
valueB:?*
dtype0*
_output_shapes
:
?
Dbert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros/ConstConst*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>bert/encoder/layer_2/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros/Const*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*

index_type0*
_output_shapes	
:?*
T0
?
,bert/encoder/layer_2/intermediate/dense/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias
?
3bert/encoder/layer_2/intermediate/dense/bias/AssignAssign,bert/encoder/layer_2/intermediate/dense/bias>bert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
1bert/encoder/layer_2/intermediate/dense/bias/readIdentity,bert/encoder/layer_2/intermediate/dense/bias*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias
?
.bert/encoder/layer_2/intermediate/dense/MatMulMatMul?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_2/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_2/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_2/intermediate/dense/MatMul1bert/encoder/layer_2/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
r
-bert/encoder/layer_2/intermediate/dense/Pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *  @@
?
+bert/encoder/layer_2/intermediate/dense/PowPow/bert/encoder/layer_2/intermediate/dense/BiasAdd-bert/encoder/layer_2/intermediate/dense/Pow/y* 
_output_shapes
:
??*
T0
r
-bert/encoder/layer_2/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_2/intermediate/dense/mulMul-bert/encoder/layer_2/intermediate/dense/mul/x+bert/encoder/layer_2/intermediate/dense/Pow*
T0* 
_output_shapes
:
??
?
+bert/encoder/layer_2/intermediate/dense/addAdd/bert/encoder/layer_2/intermediate/dense/BiasAdd+bert/encoder/layer_2/intermediate/dense/mul* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_2/intermediate/dense/mul_1/xConst*
_output_shapes
: *
valueB
 **BL?*
dtype0
?
-bert/encoder/layer_2/intermediate/dense/mul_1Mul/bert/encoder/layer_2/intermediate/dense/mul_1/x+bert/encoder/layer_2/intermediate/dense/add*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_2/intermediate/dense/TanhTanh-bert/encoder/layer_2/intermediate/dense/mul_1*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_2/intermediate/dense/add_1/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
-bert/encoder/layer_2/intermediate/dense/add_1Add/bert/encoder/layer_2/intermediate/dense/add_1/x,bert/encoder/layer_2/intermediate/dense/Tanh*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_2/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_2/intermediate/dense/mul_2Mul/bert/encoder/layer_2/intermediate/dense/mul_2/x-bert/encoder/layer_2/intermediate/dense/add_1*
T0* 
_output_shapes
:
??
?
-bert/encoder/layer_2/intermediate/dense/mul_3Mul/bert/encoder/layer_2/intermediate/dense/BiasAdd-bert/encoder/layer_2/intermediate/dense/mul_2*
T0* 
_output_shapes
:
??
?
Kbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Jbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Lbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
valueB
 *
ף<*
dtype0
?
Ubert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
seed2 
?
Ibert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel* 
_output_shapes
:
??
?
Ebert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel
?
(bert/encoder/layer_2/output/dense/kernel
VariableV2*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
/bert/encoder/layer_2/output/dense/kernel/AssignAssign(bert/encoder/layer_2/output/dense/kernelEbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
-bert/encoder/layer_2/output/dense/kernel/readIdentity(bert/encoder/layer_2/output/dense/kernel* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel
?
8bert/encoder/layer_2/output/dense/bias/Initializer/zerosConst*9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
&bert/encoder/layer_2/output/dense/bias
VariableV2*
shared_name *9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
-bert/encoder/layer_2/output/dense/bias/AssignAssign&bert/encoder/layer_2/output/dense/bias8bert/encoder/layer_2/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
+bert/encoder/layer_2/output/dense/bias/readIdentity&bert/encoder/layer_2/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
_output_shapes	
:?
?
(bert/encoder/layer_2/output/dense/MatMulMatMul-bert/encoder/layer_2/intermediate/dense/mul_3-bert/encoder/layer_2/output/dense/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
)bert/encoder/layer_2/output/dense/BiasAddBiasAdd(bert/encoder/layer_2/output/dense/MatMul+bert/encoder/layer_2/output/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
bert/encoder/layer_2/output/addAdd)bert/encoder/layer_2/output/dense/BiasAdd?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
??*
T0
?
<bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
valueB?*    
?
*bert/encoder/layer_2/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
	container *
shape:?
?
1bert/encoder/layer_2/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_2/output/LayerNorm/beta<bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zeros*=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
/bert/encoder/layer_2/output/LayerNorm/beta/readIdentity*bert/encoder/layer_2/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
_output_shapes	
:?
?
<bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
+bert/encoder/layer_2/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
	container *
shape:?
?
2bert/encoder/layer_2/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_2/output/LayerNorm/gamma<bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
0bert/encoder/layer_2/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_2/output/LayerNorm/gamma*
_output_shapes	
:?*
T0*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma
?
Dbert/encoder/layer_2/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
?
2bert/encoder/layer_2/output/LayerNorm/moments/meanMeanbert/encoder/layer_2/output/addDbert/encoder/layer_2/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
:bert/encoder/layer_2/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_2/output/LayerNorm/moments/mean*
_output_shapes
:	?*
T0
?
?bert/encoder/layer_2/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_2/output/add:bert/encoder/layer_2/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Hbert/encoder/layer_2/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
6bert/encoder/layer_2/output/LayerNorm/moments/varianceMean?bert/encoder/layer_2/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_2/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
z
5bert/encoder/layer_2/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
3bert/encoder/layer_2/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_2/output/LayerNorm/moments/variance5bert/encoder/layer_2/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
5bert/encoder/layer_2/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_2/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
3bert/encoder/layer_2/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_2/output/LayerNorm/gamma/read* 
_output_shapes
:
??*
T0
?
5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_2/output/add3bert/encoder/layer_2/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_2/output/LayerNorm/moments/mean3bert/encoder/layer_2/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
3bert/encoder/layer_2/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_2/output/LayerNorm/beta/read5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_2/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Sbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
valueB"      
?
Rbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
seed2 
?
Qbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_3/attention/self/query/kernel
VariableV2*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
7bert/encoder/layer_3/attention/self/query/kernel/AssignAssign0bert/encoder/layer_3/attention/self/query/kernelMbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_3/attention/self/query/kernel/readIdentity0bert/encoder/layer_3/attention/self/query/kernel* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel
?
@bert/encoder/layer_3/attention/self/query/bias/Initializer/zerosConst*
_output_shapes	
:?*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
valueB?*    *
dtype0
?
.bert/encoder/layer_3/attention/self/query/bias
VariableV2*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name 
?
5bert/encoder/layer_3/attention/self/query/bias/AssignAssign.bert/encoder/layer_3/attention/self/query/bias@bert/encoder/layer_3/attention/self/query/bias/Initializer/zeros*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
3bert/encoder/layer_3/attention/self/query/bias/readIdentity.bert/encoder/layer_3/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
_output_shapes	
:?
?
0bert/encoder/layer_3/attention/self/query/MatMulMatMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_15bert/encoder/layer_3/attention/self/query/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
1bert/encoder/layer_3/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_3/attention/self/query/MatMul3bert/encoder/layer_3/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Qbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
valueB
 *    
?
Rbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
seed2 
?
Obert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_3/attention/self/key/kernel
VariableV2*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
5bert/encoder/layer_3/attention/self/key/kernel/AssignAssign.bert/encoder/layer_3/attention/self/key/kernelKbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
3bert/encoder/layer_3/attention/self/key/kernel/readIdentity.bert/encoder/layer_3/attention/self/key/kernel* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel
?
>bert/encoder/layer_3/attention/self/key/bias/Initializer/zerosConst*
_output_shapes	
:?*?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias*
valueB?*    *
dtype0
?
,bert/encoder/layer_3/attention/self/key/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias
?
3bert/encoder/layer_3/attention/self/key/bias/AssignAssign,bert/encoder/layer_3/attention/self/key/bias>bert/encoder/layer_3/attention/self/key/bias/Initializer/zeros*
T0*?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
1bert/encoder/layer_3/attention/self/key/bias/readIdentity,bert/encoder/layer_3/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias*
_output_shapes	
:?
?
.bert/encoder/layer_3/attention/self/key/MatMulMatMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_13bert/encoder/layer_3/attention/self/key/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_3/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_3/attention/self/key/MatMul1bert/encoder/layer_3/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Sbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
seed2 
?
Qbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_3/attention/self/value/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
7bert/encoder/layer_3/attention/self/value/kernel/AssignAssign0bert/encoder/layer_3/attention/self/value/kernelMbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_3/attention/self/value/kernel/readIdentity0bert/encoder/layer_3/attention/self/value/kernel* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel
?
@bert/encoder/layer_3/attention/self/value/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
valueB?*    
?
.bert/encoder/layer_3/attention/self/value/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_3/attention/self/value/bias/AssignAssign.bert/encoder/layer_3/attention/self/value/bias@bert/encoder/layer_3/attention/self/value/bias/Initializer/zeros*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
3bert/encoder/layer_3/attention/self/value/bias/readIdentity.bert/encoder/layer_3/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
_output_shapes	
:?
?
0bert/encoder/layer_3/attention/self/value/MatMulMatMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_15bert/encoder/layer_3/attention/self/value/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
1bert/encoder/layer_3/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_3/attention/self/value/MatMul3bert/encoder/layer_3/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
1bert/encoder/layer_3/attention/self/Reshape/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
+bert/encoder/layer_3/attention/self/ReshapeReshape1bert/encoder/layer_3/attention/self/query/BiasAdd1bert/encoder/layer_3/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
2bert/encoder/layer_3/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_3/attention/self/transpose	Transpose+bert/encoder/layer_3/attention/self/Reshape2bert/encoder/layer_3/attention/self/transpose/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_3/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_3/attention/self/Reshape_1Reshape/bert/encoder/layer_3/attention/self/key/BiasAdd3bert/encoder/layer_3/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_3/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_3/attention/self/transpose_1	Transpose-bert/encoder/layer_3/attention/self/Reshape_14bert/encoder/layer_3/attention/self/transpose_1/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
*bert/encoder/layer_3/attention/self/MatMulBatchMatMul-bert/encoder/layer_3/attention/self/transpose/bert/encoder/layer_3/attention/self/transpose_1*
T0*(
_output_shapes
:??*
adj_x( *
adj_y(
n
)bert/encoder/layer_3/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_3/attention/self/MulMul*bert/encoder/layer_3/attention/self/MatMul)bert/encoder/layer_3/attention/self/Mul/y*
T0*(
_output_shapes
:??
|
2bert/encoder/layer_3/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
.bert/encoder/layer_3/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_3/attention/self/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:??
n
)bert/encoder/layer_3/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_3/attention/self/subSub)bert/encoder/layer_3/attention/self/sub/x.bert/encoder/layer_3/attention/self/ExpandDims*
T0*(
_output_shapes
:??
p
+bert/encoder/layer_3/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
)bert/encoder/layer_3/attention/self/mul_1Mul'bert/encoder/layer_3/attention/self/sub+bert/encoder/layer_3/attention/self/mul_1/y*(
_output_shapes
:??*
T0
?
'bert/encoder/layer_3/attention/self/addAdd'bert/encoder/layer_3/attention/self/Mul)bert/encoder/layer_3/attention/self/mul_1*
T0*(
_output_shapes
:??
?
+bert/encoder/layer_3/attention/self/SoftmaxSoftmax'bert/encoder/layer_3/attention/self/add*
T0*(
_output_shapes
:??
?
3bert/encoder/layer_3/attention/self/Reshape_2/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_3/attention/self/Reshape_2Reshape1bert/encoder/layer_3/attention/self/value/BiasAdd3bert/encoder/layer_3/attention/self/Reshape_2/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
4bert/encoder/layer_3/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_3/attention/self/transpose_2	Transpose-bert/encoder/layer_3/attention/self/Reshape_24bert/encoder/layer_3/attention/self/transpose_2/perm*
Tperm0*
T0*'
_output_shapes
:?@
?
,bert/encoder/layer_3/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_3/attention/self/Softmax/bert/encoder/layer_3/attention/self/transpose_2*'
_output_shapes
:?@*
adj_x( *
adj_y( *
T0
?
4bert/encoder/layer_3/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_3/attention/self/transpose_3	Transpose,bert/encoder/layer_3/attention/self/MatMul_14bert/encoder/layer_3/attention/self/transpose_3/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
3bert/encoder/layer_3/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_3/attention/self/Reshape_3Reshape/bert/encoder/layer_3/attention/self/transpose_33bert/encoder/layer_3/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:
??
?
Ubert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Tbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
_bert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/shape*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??
?
Sbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel* 
_output_shapes
:
??
?
2bert/encoder/layer_3/attention/output/dense/kernel
VariableV2*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel
?
9bert/encoder/layer_3/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_3/attention/output/dense/kernelObert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
7bert/encoder/layer_3/attention/output/dense/kernel/readIdentity2bert/encoder/layer_3/attention/output/dense/kernel* 
_output_shapes
:
??*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel
?
Bbert/encoder/layer_3/attention/output/dense/bias/Initializer/zerosConst*C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
0bert/encoder/layer_3/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
	container *
shape:?
?
7bert/encoder/layer_3/attention/output/dense/bias/AssignAssign0bert/encoder/layer_3/attention/output/dense/biasBbert/encoder/layer_3/attention/output/dense/bias/Initializer/zeros*C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
5bert/encoder/layer_3/attention/output/dense/bias/readIdentity0bert/encoder/layer_3/attention/output/dense/bias*C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
_output_shapes	
:?*
T0
?
2bert/encoder/layer_3/attention/output/dense/MatMulMatMul-bert/encoder/layer_3/attention/self/Reshape_37bert/encoder/layer_3/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
3bert/encoder/layer_3/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_3/attention/output/dense/MatMul5bert/encoder/layer_3/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
)bert/encoder/layer_3/attention/output/addAdd3bert/encoder/layer_3/attention/output/dense/BiasAdd5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
Fbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zerosConst*G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
4bert/encoder/layer_3/attention/output/LayerNorm/beta
VariableV2*
shared_name *G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
;bert/encoder/layer_3/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_3/attention/output/LayerNorm/betaFbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
9bert/encoder/layer_3/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_3/attention/output/LayerNorm/beta*G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
_output_shapes	
:?*
T0
?
Fbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
valueB?*  ??
?
5bert/encoder/layer_3/attention/output/LayerNorm/gamma
VariableV2*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name 
?
<bert/encoder/layer_3/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_3/attention/output/LayerNorm/gammaFbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
:bert/encoder/layer_3/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_3/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
_output_shapes	
:?
?
Nbert/encoder/layer_3/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
<bert/encoder/layer_3/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_3/attention/output/addNbert/encoder/layer_3/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
Dbert/encoder/layer_3/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_3/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
Ibert/encoder/layer_3/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_3/attention/output/addDbert/encoder/layer_3/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Rbert/encoder/layer_3/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
@bert/encoder/layer_3/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_3/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_3/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_3/attention/output/LayerNorm/moments/variance?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	?*
T0
?
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add*
_output_shapes
:	?*
T0
?
=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_3/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_3/attention/output/add=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_3/attention/output/LayerNorm/moments/mean=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_3/attention/output/LayerNorm/beta/read?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_2* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Qbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Obert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel
?
Kbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_3/intermediate/dense/kernel
VariableV2*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel
?
5bert/encoder/layer_3/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_3/intermediate/dense/kernelKbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
3bert/encoder/layer_3/intermediate/dense/kernel/readIdentity.bert/encoder/layer_3/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
valueB:?
?
Dbert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros/ConstConst*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>bert/encoder/layer_3/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros/Const*
T0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*

index_type0*
_output_shapes	
:?
?
,bert/encoder/layer_3/intermediate/dense/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias
?
3bert/encoder/layer_3/intermediate/dense/bias/AssignAssign,bert/encoder/layer_3/intermediate/dense/bias>bert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias
?
1bert/encoder/layer_3/intermediate/dense/bias/readIdentity,bert/encoder/layer_3/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
_output_shapes	
:?
?
.bert/encoder/layer_3/intermediate/dense/MatMulMatMul?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_3/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_3/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_3/intermediate/dense/MatMul1bert/encoder/layer_3/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
r
-bert/encoder/layer_3/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_3/intermediate/dense/PowPow/bert/encoder/layer_3/intermediate/dense/BiasAdd-bert/encoder/layer_3/intermediate/dense/Pow/y*
T0* 
_output_shapes
:
??
r
-bert/encoder/layer_3/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_3/intermediate/dense/mulMul-bert/encoder/layer_3/intermediate/dense/mul/x+bert/encoder/layer_3/intermediate/dense/Pow*
T0* 
_output_shapes
:
??
?
+bert/encoder/layer_3/intermediate/dense/addAdd/bert/encoder/layer_3/intermediate/dense/BiasAdd+bert/encoder/layer_3/intermediate/dense/mul*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_3/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_3/intermediate/dense/mul_1Mul/bert/encoder/layer_3/intermediate/dense/mul_1/x+bert/encoder/layer_3/intermediate/dense/add*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_3/intermediate/dense/TanhTanh-bert/encoder/layer_3/intermediate/dense/mul_1*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_3/intermediate/dense/add_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_3/intermediate/dense/add_1Add/bert/encoder/layer_3/intermediate/dense/add_1/x,bert/encoder/layer_3/intermediate/dense/Tanh*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_3/intermediate/dense/mul_2/xConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
?
-bert/encoder/layer_3/intermediate/dense/mul_2Mul/bert/encoder/layer_3/intermediate/dense/mul_2/x-bert/encoder/layer_3/intermediate/dense/add_1*
T0* 
_output_shapes
:
??
?
-bert/encoder/layer_3/intermediate/dense/mul_3Mul/bert/encoder/layer_3/intermediate/dense/BiasAdd-bert/encoder/layer_3/intermediate/dense/mul_2*
T0* 
_output_shapes
:
??
?
Kbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Jbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Lbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
valueB
 *
ף<*
dtype0
?
Ubert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
seed2 
?
Ibert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel
?
Ebert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel
?
(bert/encoder/layer_3/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
	container *
shape:
??
?
/bert/encoder/layer_3/output/dense/kernel/AssignAssign(bert/encoder/layer_3/output/dense/kernelEbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel
?
-bert/encoder/layer_3/output/dense/kernel/readIdentity(bert/encoder/layer_3/output/dense/kernel* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel
?
8bert/encoder/layer_3/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
valueB?*    
?
&bert/encoder/layer_3/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
	container *
shape:?
?
-bert/encoder/layer_3/output/dense/bias/AssignAssign&bert/encoder/layer_3/output/dense/bias8bert/encoder/layer_3/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
+bert/encoder/layer_3/output/dense/bias/readIdentity&bert/encoder/layer_3/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
_output_shapes	
:?
?
(bert/encoder/layer_3/output/dense/MatMulMatMul-bert/encoder/layer_3/intermediate/dense/mul_3-bert/encoder/layer_3/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
)bert/encoder/layer_3/output/dense/BiasAddBiasAdd(bert/encoder/layer_3/output/dense/MatMul+bert/encoder/layer_3/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
bert/encoder/layer_3/output/addAdd)bert/encoder/layer_3/output/dense/BiasAdd?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
<bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
*bert/encoder/layer_3/output/LayerNorm/beta
VariableV2*
shared_name *=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
1bert/encoder/layer_3/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_3/output/LayerNorm/beta<bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
/bert/encoder/layer_3/output/LayerNorm/beta/readIdentity*bert/encoder/layer_3/output/LayerNorm/beta*
_output_shapes	
:?*
T0*=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta
?
<bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
+bert/encoder/layer_3/output/LayerNorm/gamma
VariableV2*
shared_name *>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
2bert/encoder/layer_3/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_3/output/LayerNorm/gamma<bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
0bert/encoder/layer_3/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_3/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
_output_shapes	
:?
?
Dbert/encoder/layer_3/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
?
2bert/encoder/layer_3/output/LayerNorm/moments/meanMeanbert/encoder/layer_3/output/addDbert/encoder/layer_3/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
:bert/encoder/layer_3/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_3/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_3/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_3/output/add:bert/encoder/layer_3/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Hbert/encoder/layer_3/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
6bert/encoder/layer_3/output/LayerNorm/moments/varianceMean?bert/encoder/layer_3/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_3/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
z
5bert/encoder/layer_3/output/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼?+
?
3bert/encoder/layer_3/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_3/output/LayerNorm/moments/variance5bert/encoder/layer_3/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	?*
T0
?
5bert/encoder/layer_3/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_3/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
3bert/encoder/layer_3/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_3/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_3/output/add3bert/encoder/layer_3/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_3/output/LayerNorm/moments/mean3bert/encoder/layer_3/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
3bert/encoder/layer_3/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_3/output/LayerNorm/beta/read5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_3/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Sbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
seed2 
?
Qbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_4/attention/self/query/kernel
VariableV2*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
7bert/encoder/layer_4/attention/self/query/kernel/AssignAssign0bert/encoder/layer_4/attention/self/query/kernelMbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_4/attention/self/query/kernel/readIdentity0bert/encoder/layer_4/attention/self/query/kernel* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel
?
@bert/encoder/layer_4/attention/self/query/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_4/attention/self/query/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias
?
5bert/encoder/layer_4/attention/self/query/bias/AssignAssign.bert/encoder/layer_4/attention/self/query/bias@bert/encoder/layer_4/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
3bert/encoder/layer_4/attention/self/query/bias/readIdentity.bert/encoder/layer_4/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
_output_shapes	
:?
?
0bert/encoder/layer_4/attention/self/query/MatMulMatMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_15bert/encoder/layer_4/attention/self/query/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
1bert/encoder/layer_4/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_4/attention/self/query/MatMul3bert/encoder/layer_4/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Qbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
valueB"      
?
Pbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
seed2 
?
Obert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_4/attention/self/key/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
	container *
shape:
??
?
5bert/encoder/layer_4/attention/self/key/kernel/AssignAssign.bert/encoder/layer_4/attention/self/key/kernelKbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel
?
3bert/encoder/layer_4/attention/self/key/kernel/readIdentity.bert/encoder/layer_4/attention/self/key/kernel*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel* 
_output_shapes
:
??*
T0
?
>bert/encoder/layer_4/attention/self/key/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
,bert/encoder/layer_4/attention/self/key/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias
?
3bert/encoder/layer_4/attention/self/key/bias/AssignAssign,bert/encoder/layer_4/attention/self/key/bias>bert/encoder/layer_4/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_4/attention/self/key/bias/readIdentity,bert/encoder/layer_4/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
_output_shapes	
:?
?
.bert/encoder/layer_4/attention/self/key/MatMulMatMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_13bert/encoder/layer_4/attention/self/key/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_4/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_4/attention/self/key/MatMul1bert/encoder/layer_4/attention/self/key/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
Sbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Qbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_4/attention/self/value/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
	container *
shape:
??
?
7bert/encoder/layer_4/attention/self/value/kernel/AssignAssign0bert/encoder/layer_4/attention/self/value/kernelMbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_4/attention/self/value/kernel/readIdentity0bert/encoder/layer_4/attention/self/value/kernel*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel* 
_output_shapes
:
??*
T0
?
@bert/encoder/layer_4/attention/self/value/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_4/attention/self/value/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
	container *
shape:?
?
5bert/encoder/layer_4/attention/self/value/bias/AssignAssign.bert/encoder/layer_4/attention/self/value/bias@bert/encoder/layer_4/attention/self/value/bias/Initializer/zeros*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
3bert/encoder/layer_4/attention/self/value/bias/readIdentity.bert/encoder/layer_4/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
_output_shapes	
:?
?
0bert/encoder/layer_4/attention/self/value/MatMulMatMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_15bert/encoder/layer_4/attention/self/value/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
1bert/encoder/layer_4/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_4/attention/self/value/MatMul3bert/encoder/layer_4/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
1bert/encoder/layer_4/attention/self/Reshape/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
+bert/encoder/layer_4/attention/self/ReshapeReshape1bert/encoder/layer_4/attention/self/query/BiasAdd1bert/encoder/layer_4/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
2bert/encoder/layer_4/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_4/attention/self/transpose	Transpose+bert/encoder/layer_4/attention/self/Reshape2bert/encoder/layer_4/attention/self/transpose/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_4/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_4/attention/self/Reshape_1Reshape/bert/encoder/layer_4/attention/self/key/BiasAdd3bert/encoder/layer_4/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_4/attention/self/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
?
/bert/encoder/layer_4/attention/self/transpose_1	Transpose-bert/encoder/layer_4/attention/self/Reshape_14bert/encoder/layer_4/attention/self/transpose_1/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
*bert/encoder/layer_4/attention/self/MatMulBatchMatMul-bert/encoder/layer_4/attention/self/transpose/bert/encoder/layer_4/attention/self/transpose_1*
adj_y(*
T0*(
_output_shapes
:??*
adj_x( 
n
)bert/encoder/layer_4/attention/self/Mul/yConst*
dtype0*
_output_shapes
: *
valueB
 *   >
?
'bert/encoder/layer_4/attention/self/MulMul*bert/encoder/layer_4/attention/self/MatMul)bert/encoder/layer_4/attention/self/Mul/y*(
_output_shapes
:??*
T0
|
2bert/encoder/layer_4/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
.bert/encoder/layer_4/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_4/attention/self/ExpandDims/dim*(
_output_shapes
:??*

Tdim0*
T0
n
)bert/encoder/layer_4/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_4/attention/self/subSub)bert/encoder/layer_4/attention/self/sub/x.bert/encoder/layer_4/attention/self/ExpandDims*
T0*(
_output_shapes
:??
p
+bert/encoder/layer_4/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
)bert/encoder/layer_4/attention/self/mul_1Mul'bert/encoder/layer_4/attention/self/sub+bert/encoder/layer_4/attention/self/mul_1/y*
T0*(
_output_shapes
:??
?
'bert/encoder/layer_4/attention/self/addAdd'bert/encoder/layer_4/attention/self/Mul)bert/encoder/layer_4/attention/self/mul_1*(
_output_shapes
:??*
T0
?
+bert/encoder/layer_4/attention/self/SoftmaxSoftmax'bert/encoder/layer_4/attention/self/add*(
_output_shapes
:??*
T0
?
3bert/encoder/layer_4/attention/self/Reshape_2/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_4/attention/self/Reshape_2Reshape1bert/encoder/layer_4/attention/self/value/BiasAdd3bert/encoder/layer_4/attention/self/Reshape_2/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
4bert/encoder/layer_4/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_4/attention/self/transpose_2	Transpose-bert/encoder/layer_4/attention/self/Reshape_24bert/encoder/layer_4/attention/self/transpose_2/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
,bert/encoder/layer_4/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_4/attention/self/Softmax/bert/encoder/layer_4/attention/self/transpose_2*
adj_x( *
adj_y( *
T0*'
_output_shapes
:?@
?
4bert/encoder/layer_4/attention/self/transpose_3/permConst*
dtype0*
_output_shapes
:*%
valueB"             
?
/bert/encoder/layer_4/attention/self/transpose_3	Transpose,bert/encoder/layer_4/attention/self/MatMul_14bert/encoder/layer_4/attention/self/transpose_3/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_4/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_4/attention/self/Reshape_3Reshape/bert/encoder/layer_4/attention/self/transpose_33bert/encoder/layer_4/attention/self/Reshape_3/shape*
Tshape0* 
_output_shapes
:
??*
T0
?
Ubert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Tbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
_bert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/shape*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Sbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel
?
Obert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel* 
_output_shapes
:
??
?
2bert/encoder/layer_4/attention/output/dense/kernel
VariableV2*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
9bert/encoder/layer_4/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_4/attention/output/dense/kernelObert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel
?
7bert/encoder/layer_4/attention/output/dense/kernel/readIdentity2bert/encoder/layer_4/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel* 
_output_shapes
:
??
?
Bbert/encoder/layer_4/attention/output/dense/bias/Initializer/zerosConst*C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
0bert/encoder/layer_4/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
	container *
shape:?
?
7bert/encoder/layer_4/attention/output/dense/bias/AssignAssign0bert/encoder/layer_4/attention/output/dense/biasBbert/encoder/layer_4/attention/output/dense/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
validate_shape(
?
5bert/encoder/layer_4/attention/output/dense/bias/readIdentity0bert/encoder/layer_4/attention/output/dense/bias*
_output_shapes	
:?*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias
?
2bert/encoder/layer_4/attention/output/dense/MatMulMatMul-bert/encoder/layer_4/attention/self/Reshape_37bert/encoder/layer_4/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
3bert/encoder/layer_4/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_4/attention/output/dense/MatMul5bert/encoder/layer_4/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
)bert/encoder/layer_4/attention/output/addAdd3bert/encoder/layer_4/attention/output/dense/BiasAdd5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
Fbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zerosConst*G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
4bert/encoder/layer_4/attention/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
	container *
shape:?
?
;bert/encoder/layer_4/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_4/attention/output/LayerNorm/betaFbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zeros*G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
9bert/encoder/layer_4/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_4/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
_output_shapes	
:?
?
Fbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_4/attention/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
	container *
shape:?
?
<bert/encoder/layer_4/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_4/attention/output/LayerNorm/gammaFbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
:bert/encoder/layer_4/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_4/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
_output_shapes	
:?
?
Nbert/encoder/layer_4/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
?
<bert/encoder/layer_4/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_4/attention/output/addNbert/encoder/layer_4/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
Dbert/encoder/layer_4/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_4/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
Ibert/encoder/layer_4/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_4/attention/output/addDbert/encoder/layer_4/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Rbert/encoder/layer_4/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
?
@bert/encoder/layer_4/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_4/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_4/attention/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼?+
?
=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_4/attention/output/LayerNorm/moments/variance?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_4/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_4/attention/output/add=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_4/attention/output/LayerNorm/moments/mean=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_4/attention/output/LayerNorm/beta/read?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_2* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Qbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
valueB
 *
ף<*
dtype0
?
[bert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Obert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel
?
.bert/encoder/layer_4/intermediate/dense/kernel
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
5bert/encoder/layer_4/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_4/intermediate/dense/kernelKbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
3bert/encoder/layer_4/intermediate/dense/kernel/readIdentity.bert/encoder/layer_4/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
valueB:?
?
Dbert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros/ConstConst*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>bert/encoder/layer_4/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros/Const*
T0*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*

index_type0*
_output_shapes	
:?
?
,bert/encoder/layer_4/intermediate/dense/bias
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
3bert/encoder/layer_4/intermediate/dense/bias/AssignAssign,bert/encoder/layer_4/intermediate/dense/bias>bert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_4/intermediate/dense/bias/readIdentity,bert/encoder/layer_4/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
_output_shapes	
:?
?
.bert/encoder/layer_4/intermediate/dense/MatMulMatMul?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_4/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_4/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_4/intermediate/dense/MatMul1bert/encoder/layer_4/intermediate/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
r
-bert/encoder/layer_4/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_4/intermediate/dense/PowPow/bert/encoder/layer_4/intermediate/dense/BiasAdd-bert/encoder/layer_4/intermediate/dense/Pow/y*
T0* 
_output_shapes
:
??
r
-bert/encoder/layer_4/intermediate/dense/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *'7=
?
+bert/encoder/layer_4/intermediate/dense/mulMul-bert/encoder/layer_4/intermediate/dense/mul/x+bert/encoder/layer_4/intermediate/dense/Pow*
T0* 
_output_shapes
:
??
?
+bert/encoder/layer_4/intermediate/dense/addAdd/bert/encoder/layer_4/intermediate/dense/BiasAdd+bert/encoder/layer_4/intermediate/dense/mul*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_4/intermediate/dense/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 **BL?
?
-bert/encoder/layer_4/intermediate/dense/mul_1Mul/bert/encoder/layer_4/intermediate/dense/mul_1/x+bert/encoder/layer_4/intermediate/dense/add*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_4/intermediate/dense/TanhTanh-bert/encoder/layer_4/intermediate/dense/mul_1* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_4/intermediate/dense/add_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_4/intermediate/dense/add_1Add/bert/encoder/layer_4/intermediate/dense/add_1/x,bert/encoder/layer_4/intermediate/dense/Tanh* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_4/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_4/intermediate/dense/mul_2Mul/bert/encoder/layer_4/intermediate/dense/mul_2/x-bert/encoder/layer_4/intermediate/dense/add_1*
T0* 
_output_shapes
:
??
?
-bert/encoder/layer_4/intermediate/dense/mul_3Mul/bert/encoder/layer_4/intermediate/dense/BiasAdd-bert/encoder/layer_4/intermediate/dense/mul_2*
T0* 
_output_shapes
:
??
?
Kbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Jbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Lbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/stddevConst*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
Ubert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/shape*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0
?
Ibert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel
?
Ebert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/mean*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel* 
_output_shapes
:
??*
T0
?
(bert/encoder/layer_4/output/dense/kernel
VariableV2*
shared_name *;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
/bert/encoder/layer_4/output/dense/kernel/AssignAssign(bert/encoder/layer_4/output/dense/kernelEbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
-bert/encoder/layer_4/output/dense/kernel/readIdentity(bert/encoder/layer_4/output/dense/kernel*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel* 
_output_shapes
:
??*
T0
?
8bert/encoder/layer_4/output/dense/bias/Initializer/zerosConst*9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
&bert/encoder/layer_4/output/dense/bias
VariableV2*
shared_name *9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
-bert/encoder/layer_4/output/dense/bias/AssignAssign&bert/encoder/layer_4/output/dense/bias8bert/encoder/layer_4/output/dense/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
validate_shape(
?
+bert/encoder/layer_4/output/dense/bias/readIdentity&bert/encoder/layer_4/output/dense/bias*9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
_output_shapes	
:?*
T0
?
(bert/encoder/layer_4/output/dense/MatMulMatMul-bert/encoder/layer_4/intermediate/dense/mul_3-bert/encoder/layer_4/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
)bert/encoder/layer_4/output/dense/BiasAddBiasAdd(bert/encoder/layer_4/output/dense/MatMul+bert/encoder/layer_4/output/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
bert/encoder/layer_4/output/addAdd)bert/encoder/layer_4/output/dense/BiasAdd?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
<bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
*bert/encoder/layer_4/output/LayerNorm/beta
VariableV2*
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
	container *
shape:?*
dtype0
?
1bert/encoder/layer_4/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_4/output/LayerNorm/beta<bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta
?
/bert/encoder/layer_4/output/LayerNorm/beta/readIdentity*bert/encoder/layer_4/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
_output_shapes	
:?
?
<bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
+bert/encoder/layer_4/output/LayerNorm/gamma
VariableV2*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name 
?
2bert/encoder/layer_4/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_4/output/LayerNorm/gamma<bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma
?
0bert/encoder/layer_4/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_4/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
_output_shapes	
:?
?
Dbert/encoder/layer_4/output/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
?
2bert/encoder/layer_4/output/LayerNorm/moments/meanMeanbert/encoder/layer_4/output/addDbert/encoder/layer_4/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
:bert/encoder/layer_4/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_4/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_4/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_4/output/add:bert/encoder/layer_4/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
??*
T0
?
Hbert/encoder/layer_4/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
?
6bert/encoder/layer_4/output/LayerNorm/moments/varianceMean?bert/encoder/layer_4/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_4/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
z
5bert/encoder/layer_4/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
3bert/encoder/layer_4/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_4/output/LayerNorm/moments/variance5bert/encoder/layer_4/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
5bert/encoder/layer_4/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_4/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
3bert/encoder/layer_4/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_4/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_4/output/LayerNorm/gamma/read* 
_output_shapes
:
??*
T0
?
5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_4/output/add3bert/encoder/layer_4/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_4/output/LayerNorm/moments/mean3bert/encoder/layer_4/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
3bert/encoder/layer_4/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_4/output/LayerNorm/beta/read5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_4/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Sbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel
?
Qbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_5/attention/self/query/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
	container *
shape:
??
?
7bert/encoder/layer_5/attention/self/query/kernel/AssignAssign0bert/encoder/layer_5/attention/self/query/kernelMbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
5bert/encoder/layer_5/attention/self/query/kernel/readIdentity0bert/encoder/layer_5/attention/self/query/kernel* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel
?
@bert/encoder/layer_5/attention/self/query/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_5/attention/self/query/bias
VariableV2*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
	container *
shape:?*
dtype0
?
5bert/encoder/layer_5/attention/self/query/bias/AssignAssign.bert/encoder/layer_5/attention/self/query/bias@bert/encoder/layer_5/attention/self/query/bias/Initializer/zeros*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
3bert/encoder/layer_5/attention/self/query/bias/readIdentity.bert/encoder/layer_5/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
_output_shapes	
:?
?
0bert/encoder/layer_5/attention/self/query/MatMulMatMul5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_15bert/encoder/layer_5/attention/self/query/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
1bert/encoder/layer_5/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_5/attention/self/query/MatMul3bert/encoder/layer_5/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Qbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/shape*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0
?
Obert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_5/attention/self/key/kernel
VariableV2*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel
?
5bert/encoder/layer_5/attention/self/key/kernel/AssignAssign.bert/encoder/layer_5/attention/self/key/kernelKbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
3bert/encoder/layer_5/attention/self/key/kernel/readIdentity.bert/encoder/layer_5/attention/self/key/kernel*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel* 
_output_shapes
:
??*
T0
?
>bert/encoder/layer_5/attention/self/key/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
,bert/encoder/layer_5/attention/self/key/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
	container *
shape:?
?
3bert/encoder/layer_5/attention/self/key/bias/AssignAssign,bert/encoder/layer_5/attention/self/key/bias>bert/encoder/layer_5/attention/self/key/bias/Initializer/zeros*
T0*?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
1bert/encoder/layer_5/attention/self/key/bias/readIdentity,bert/encoder/layer_5/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
_output_shapes	
:?
?
.bert/encoder/layer_5/attention/self/key/MatMulMatMul5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_13bert/encoder/layer_5/attention/self/key/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
/bert/encoder/layer_5/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_5/attention/self/key/MatMul1bert/encoder/layer_5/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Sbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
valueB"      *
dtype0
?
Rbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
seed2 
?
Qbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_5/attention/self/value/kernel
VariableV2*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel
?
7bert/encoder/layer_5/attention/self/value/kernel/AssignAssign0bert/encoder/layer_5/attention/self/value/kernelMbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
5bert/encoder/layer_5/attention/self/value/kernel/readIdentity0bert/encoder/layer_5/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_5/attention/self/value/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_5/attention/self/value/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_5/attention/self/value/bias/AssignAssign.bert/encoder/layer_5/attention/self/value/bias@bert/encoder/layer_5/attention/self/value/bias/Initializer/zeros*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
3bert/encoder/layer_5/attention/self/value/bias/readIdentity.bert/encoder/layer_5/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias*
_output_shapes	
:?
?
0bert/encoder/layer_5/attention/self/value/MatMulMatMul5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_15bert/encoder/layer_5/attention/self/value/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
1bert/encoder/layer_5/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_5/attention/self/value/MatMul3bert/encoder/layer_5/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
1bert/encoder/layer_5/attention/self/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?      @   
?
+bert/encoder/layer_5/attention/self/ReshapeReshape1bert/encoder/layer_5/attention/self/query/BiasAdd1bert/encoder/layer_5/attention/self/Reshape/shape*
Tshape0*'
_output_shapes
:?@*
T0
?
2bert/encoder/layer_5/attention/self/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
?
-bert/encoder/layer_5/attention/self/transpose	Transpose+bert/encoder/layer_5/attention/self/Reshape2bert/encoder/layer_5/attention/self/transpose/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_5/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_5/attention/self/Reshape_1Reshape/bert/encoder/layer_5/attention/self/key/BiasAdd3bert/encoder/layer_5/attention/self/Reshape_1/shape*
Tshape0*'
_output_shapes
:?@*
T0
?
4bert/encoder/layer_5/attention/self/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
?
/bert/encoder/layer_5/attention/self/transpose_1	Transpose-bert/encoder/layer_5/attention/self/Reshape_14bert/encoder/layer_5/attention/self/transpose_1/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
*bert/encoder/layer_5/attention/self/MatMulBatchMatMul-bert/encoder/layer_5/attention/self/transpose/bert/encoder/layer_5/attention/self/transpose_1*(
_output_shapes
:??*
adj_x( *
adj_y(*
T0
n
)bert/encoder/layer_5/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_5/attention/self/MulMul*bert/encoder/layer_5/attention/self/MatMul)bert/encoder/layer_5/attention/self/Mul/y*(
_output_shapes
:??*
T0
|
2bert/encoder/layer_5/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
.bert/encoder/layer_5/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_5/attention/self/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:??
n
)bert/encoder/layer_5/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_5/attention/self/subSub)bert/encoder/layer_5/attention/self/sub/x.bert/encoder/layer_5/attention/self/ExpandDims*
T0*(
_output_shapes
:??
p
+bert/encoder/layer_5/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
)bert/encoder/layer_5/attention/self/mul_1Mul'bert/encoder/layer_5/attention/self/sub+bert/encoder/layer_5/attention/self/mul_1/y*
T0*(
_output_shapes
:??
?
'bert/encoder/layer_5/attention/self/addAdd'bert/encoder/layer_5/attention/self/Mul)bert/encoder/layer_5/attention/self/mul_1*
T0*(
_output_shapes
:??
?
+bert/encoder/layer_5/attention/self/SoftmaxSoftmax'bert/encoder/layer_5/attention/self/add*(
_output_shapes
:??*
T0
?
3bert/encoder/layer_5/attention/self/Reshape_2/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_5/attention/self/Reshape_2Reshape1bert/encoder/layer_5/attention/self/value/BiasAdd3bert/encoder/layer_5/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_5/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_5/attention/self/transpose_2	Transpose-bert/encoder/layer_5/attention/self/Reshape_24bert/encoder/layer_5/attention/self/transpose_2/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
,bert/encoder/layer_5/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_5/attention/self/Softmax/bert/encoder/layer_5/attention/self/transpose_2*
adj_x( *
adj_y( *
T0*'
_output_shapes
:?@
?
4bert/encoder/layer_5/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_5/attention/self/transpose_3	Transpose,bert/encoder/layer_5/attention/self/MatMul_14bert/encoder/layer_5/attention/self/transpose_3/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_5/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_5/attention/self/Reshape_3Reshape/bert/encoder/layer_5/attention/self/transpose_33bert/encoder/layer_5/attention/self/Reshape_3/shape* 
_output_shapes
:
??*
T0*
Tshape0
?
Ubert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
valueB"      *
dtype0
?
Tbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
_bert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
seed2 
?
Sbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel* 
_output_shapes
:
??
?
2bert/encoder/layer_5/attention/output/dense/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
	container 
?
9bert/encoder/layer_5/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_5/attention/output/dense/kernelObert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
7bert/encoder/layer_5/attention/output/dense/kernel/readIdentity2bert/encoder/layer_5/attention/output/dense/kernel* 
_output_shapes
:
??*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel
?
Bbert/encoder/layer_5/attention/output/dense/bias/Initializer/zerosConst*C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
0bert/encoder/layer_5/attention/output/dense/bias
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
7bert/encoder/layer_5/attention/output/dense/bias/AssignAssign0bert/encoder/layer_5/attention/output/dense/biasBbert/encoder/layer_5/attention/output/dense/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
5bert/encoder/layer_5/attention/output/dense/bias/readIdentity0bert/encoder/layer_5/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
_output_shapes	
:?
?
2bert/encoder/layer_5/attention/output/dense/MatMulMatMul-bert/encoder/layer_5/attention/self/Reshape_37bert/encoder/layer_5/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
3bert/encoder/layer_5/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_5/attention/output/dense/MatMul5bert/encoder/layer_5/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
)bert/encoder/layer_5/attention/output/addAdd3bert/encoder/layer_5/attention/output/dense/BiasAdd5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
Fbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zerosConst*G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
4bert/encoder/layer_5/attention/output/LayerNorm/beta
VariableV2*
_output_shapes	
:?*
shared_name *G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
	container *
shape:?*
dtype0
?
;bert/encoder/layer_5/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_5/attention/output/LayerNorm/betaFbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
9bert/encoder/layer_5/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_5/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
_output_shapes	
:?
?
Fbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_5/attention/output/LayerNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma
?
<bert/encoder/layer_5/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_5/attention/output/LayerNorm/gammaFbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
:bert/encoder/layer_5/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_5/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
_output_shapes	
:?
?
Nbert/encoder/layer_5/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
<bert/encoder/layer_5/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_5/attention/output/addNbert/encoder/layer_5/attention/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
Dbert/encoder/layer_5/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_5/attention/output/LayerNorm/moments/mean*
_output_shapes
:	?*
T0
?
Ibert/encoder/layer_5/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_5/attention/output/addDbert/encoder/layer_5/attention/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
??*
T0
?
Rbert/encoder/layer_5/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
?
@bert/encoder/layer_5/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_5/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_5/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_5/attention/output/LayerNorm/moments/variance?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_5/attention/output/LayerNorm/gamma/read* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_5/attention/output/add=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_5/attention/output/LayerNorm/moments/mean=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_5/attention/output/LayerNorm/beta/read?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Qbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
valueB"      
?
Pbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
valueB
 *
ף<
?
[bert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
seed2 
?
Obert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_5/intermediate/dense/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
	container 
?
5bert/encoder/layer_5/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_5/intermediate/dense/kernelKbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
3bert/encoder/layer_5/intermediate/dense/kernel/readIdentity.bert/encoder/layer_5/intermediate/dense/kernel* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel
?
Nbert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
valueB:?*
dtype0*
_output_shapes
:
?
Dbert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros/ConstConst*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>bert/encoder/layer_5/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros/Const*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*

index_type0
?
,bert/encoder/layer_5/intermediate/dense/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias
?
3bert/encoder/layer_5/intermediate/dense/bias/AssignAssign,bert/encoder/layer_5/intermediate/dense/bias>bert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_5/intermediate/dense/bias/readIdentity,bert/encoder/layer_5/intermediate/dense/bias*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias
?
.bert/encoder/layer_5/intermediate/dense/MatMulMatMul?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_5/intermediate/dense/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
/bert/encoder/layer_5/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_5/intermediate/dense/MatMul1bert/encoder/layer_5/intermediate/dense/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
r
-bert/encoder/layer_5/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_5/intermediate/dense/PowPow/bert/encoder/layer_5/intermediate/dense/BiasAdd-bert/encoder/layer_5/intermediate/dense/Pow/y*
T0* 
_output_shapes
:
??
r
-bert/encoder/layer_5/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_5/intermediate/dense/mulMul-bert/encoder/layer_5/intermediate/dense/mul/x+bert/encoder/layer_5/intermediate/dense/Pow*
T0* 
_output_shapes
:
??
?
+bert/encoder/layer_5/intermediate/dense/addAdd/bert/encoder/layer_5/intermediate/dense/BiasAdd+bert/encoder/layer_5/intermediate/dense/mul*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_5/intermediate/dense/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 **BL?
?
-bert/encoder/layer_5/intermediate/dense/mul_1Mul/bert/encoder/layer_5/intermediate/dense/mul_1/x+bert/encoder/layer_5/intermediate/dense/add*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_5/intermediate/dense/TanhTanh-bert/encoder/layer_5/intermediate/dense/mul_1* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_5/intermediate/dense/add_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
-bert/encoder/layer_5/intermediate/dense/add_1Add/bert/encoder/layer_5/intermediate/dense/add_1/x,bert/encoder/layer_5/intermediate/dense/Tanh* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_5/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_5/intermediate/dense/mul_2Mul/bert/encoder/layer_5/intermediate/dense/mul_2/x-bert/encoder/layer_5/intermediate/dense/add_1* 
_output_shapes
:
??*
T0
?
-bert/encoder/layer_5/intermediate/dense/mul_3Mul/bert/encoder/layer_5/intermediate/dense/BiasAdd-bert/encoder/layer_5/intermediate/dense/mul_2* 
_output_shapes
:
??*
T0
?
Kbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Jbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Lbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
valueB
 *
ף<
?
Ubert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
seed2 
?
Ibert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel* 
_output_shapes
:
??
?
Ebert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel* 
_output_shapes
:
??
?
(bert/encoder/layer_5/output/dense/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
	container 
?
/bert/encoder/layer_5/output/dense/kernel/AssignAssign(bert/encoder/layer_5/output/dense/kernelEbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
-bert/encoder/layer_5/output/dense/kernel/readIdentity(bert/encoder/layer_5/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel* 
_output_shapes
:
??
?
8bert/encoder/layer_5/output/dense/bias/Initializer/zerosConst*9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
&bert/encoder/layer_5/output/dense/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias*
	container 
?
-bert/encoder/layer_5/output/dense/bias/AssignAssign&bert/encoder/layer_5/output/dense/bias8bert/encoder/layer_5/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
+bert/encoder/layer_5/output/dense/bias/readIdentity&bert/encoder/layer_5/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias*
_output_shapes	
:?
?
(bert/encoder/layer_5/output/dense/MatMulMatMul-bert/encoder/layer_5/intermediate/dense/mul_3-bert/encoder/layer_5/output/dense/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
)bert/encoder/layer_5/output/dense/BiasAddBiasAdd(bert/encoder/layer_5/output/dense/MatMul+bert/encoder/layer_5/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
bert/encoder/layer_5/output/addAdd)bert/encoder/layer_5/output/dense/BiasAdd?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
??*
T0
?
<bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
*bert/encoder/layer_5/output/LayerNorm/beta
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
	container 
?
1bert/encoder/layer_5/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_5/output/LayerNorm/beta<bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
/bert/encoder/layer_5/output/LayerNorm/beta/readIdentity*bert/encoder/layer_5/output/LayerNorm/beta*
_output_shapes	
:?*
T0*=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta
?
<bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
+bert/encoder/layer_5/output/LayerNorm/gamma
VariableV2*
shared_name *>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
2bert/encoder/layer_5/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_5/output/LayerNorm/gamma<bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
0bert/encoder/layer_5/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_5/output/LayerNorm/gamma*
_output_shapes	
:?*
T0*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma
?
Dbert/encoder/layer_5/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
2bert/encoder/layer_5/output/LayerNorm/moments/meanMeanbert/encoder/layer_5/output/addDbert/encoder/layer_5/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
:bert/encoder/layer_5/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_5/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_5/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_5/output/add:bert/encoder/layer_5/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Hbert/encoder/layer_5/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
?
6bert/encoder/layer_5/output/LayerNorm/moments/varianceMean?bert/encoder/layer_5/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_5/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
z
5bert/encoder/layer_5/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
3bert/encoder/layer_5/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_5/output/LayerNorm/moments/variance5bert/encoder/layer_5/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	?*
T0
?
5bert/encoder/layer_5/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_5/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
3bert/encoder/layer_5/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_5/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_5/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_5/output/add3bert/encoder/layer_5/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_5/output/LayerNorm/moments/mean3bert/encoder/layer_5/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
3bert/encoder/layer_5/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_5/output/LayerNorm/beta/read5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_5/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
??*
T0
?
Sbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
valueB"      *
dtype0
?
Rbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
valueB
 *
ף<*
dtype0
?
]bert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
seed2 *
dtype0
?
Qbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_6/attention/self/query/kernel
VariableV2*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
7bert/encoder/layer_6/attention/self/query/kernel/AssignAssign0bert/encoder/layer_6/attention/self/query/kernelMbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_6/attention/self/query/kernel/readIdentity0bert/encoder/layer_6/attention/self/query/kernel*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel* 
_output_shapes
:
??*
T0
?
@bert/encoder/layer_6/attention/self/query/bias/Initializer/zerosConst*
_output_shapes	
:?*A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
valueB?*    *
dtype0
?
.bert/encoder/layer_6/attention/self/query/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_6/attention/self/query/bias/AssignAssign.bert/encoder/layer_6/attention/self/query/bias@bert/encoder/layer_6/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
3bert/encoder/layer_6/attention/self/query/bias/readIdentity.bert/encoder/layer_6/attention/self/query/bias*
_output_shapes	
:?*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias
?
0bert/encoder/layer_6/attention/self/query/MatMulMatMul5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_15bert/encoder/layer_6/attention/self/query/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
1bert/encoder/layer_6/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_6/attention/self/query/MatMul3bert/encoder/layer_6/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Qbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
valueB
 *    *
dtype0
?
Rbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
valueB
 *
ף<
?
[bert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
seed2 
?
Obert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_6/attention/self/key/kernel
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
5bert/encoder/layer_6/attention/self/key/kernel/AssignAssign.bert/encoder/layer_6/attention/self/key/kernelKbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
3bert/encoder/layer_6/attention/self/key/kernel/readIdentity.bert/encoder/layer_6/attention/self/key/kernel* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel
?
>bert/encoder/layer_6/attention/self/key/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
,bert/encoder/layer_6/attention/self/key/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias
?
3bert/encoder/layer_6/attention/self/key/bias/AssignAssign,bert/encoder/layer_6/attention/self/key/bias>bert/encoder/layer_6/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_6/attention/self/key/bias/readIdentity,bert/encoder/layer_6/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias*
_output_shapes	
:?
?
.bert/encoder/layer_6/attention/self/key/MatMulMatMul5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_13bert/encoder/layer_6/attention/self/key/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
/bert/encoder/layer_6/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_6/attention/self/key/MatMul1bert/encoder/layer_6/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Sbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Qbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_6/attention/self/value/kernel
VariableV2* 
_output_shapes
:
??*
shared_name *C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
	container *
shape:
??*
dtype0
?
7bert/encoder/layer_6/attention/self/value/kernel/AssignAssign0bert/encoder/layer_6/attention/self/value/kernelMbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_6/attention/self/value/kernel/readIdentity0bert/encoder/layer_6/attention/self/value/kernel*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel* 
_output_shapes
:
??*
T0
?
@bert/encoder/layer_6/attention/self/value/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_6/attention/self/value/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_6/attention/self/value/bias/AssignAssign.bert/encoder/layer_6/attention/self/value/bias@bert/encoder/layer_6/attention/self/value/bias/Initializer/zeros*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
3bert/encoder/layer_6/attention/self/value/bias/readIdentity.bert/encoder/layer_6/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
_output_shapes	
:?
?
0bert/encoder/layer_6/attention/self/value/MatMulMatMul5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_15bert/encoder/layer_6/attention/self/value/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
1bert/encoder/layer_6/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_6/attention/self/value/MatMul3bert/encoder/layer_6/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
1bert/encoder/layer_6/attention/self/Reshape/shapeConst*
_output_shapes
:*%
valueB"   ?      @   *
dtype0
?
+bert/encoder/layer_6/attention/self/ReshapeReshape1bert/encoder/layer_6/attention/self/query/BiasAdd1bert/encoder/layer_6/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
2bert/encoder/layer_6/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_6/attention/self/transpose	Transpose+bert/encoder/layer_6/attention/self/Reshape2bert/encoder/layer_6/attention/self/transpose/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_6/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_6/attention/self/Reshape_1Reshape/bert/encoder/layer_6/attention/self/key/BiasAdd3bert/encoder/layer_6/attention/self/Reshape_1/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
4bert/encoder/layer_6/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_6/attention/self/transpose_1	Transpose-bert/encoder/layer_6/attention/self/Reshape_14bert/encoder/layer_6/attention/self/transpose_1/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
*bert/encoder/layer_6/attention/self/MatMulBatchMatMul-bert/encoder/layer_6/attention/self/transpose/bert/encoder/layer_6/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:??
n
)bert/encoder/layer_6/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_6/attention/self/MulMul*bert/encoder/layer_6/attention/self/MatMul)bert/encoder/layer_6/attention/self/Mul/y*
T0*(
_output_shapes
:??
|
2bert/encoder/layer_6/attention/self/ExpandDims/dimConst*
dtype0*
_output_shapes
:*
valueB:
?
.bert/encoder/layer_6/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_6/attention/self/ExpandDims/dim*(
_output_shapes
:??*

Tdim0*
T0
n
)bert/encoder/layer_6/attention/self/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
'bert/encoder/layer_6/attention/self/subSub)bert/encoder/layer_6/attention/self/sub/x.bert/encoder/layer_6/attention/self/ExpandDims*(
_output_shapes
:??*
T0
p
+bert/encoder/layer_6/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
)bert/encoder/layer_6/attention/self/mul_1Mul'bert/encoder/layer_6/attention/self/sub+bert/encoder/layer_6/attention/self/mul_1/y*
T0*(
_output_shapes
:??
?
'bert/encoder/layer_6/attention/self/addAdd'bert/encoder/layer_6/attention/self/Mul)bert/encoder/layer_6/attention/self/mul_1*
T0*(
_output_shapes
:??
?
+bert/encoder/layer_6/attention/self/SoftmaxSoftmax'bert/encoder/layer_6/attention/self/add*
T0*(
_output_shapes
:??
?
3bert/encoder/layer_6/attention/self/Reshape_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?      @   
?
-bert/encoder/layer_6/attention/self/Reshape_2Reshape1bert/encoder/layer_6/attention/self/value/BiasAdd3bert/encoder/layer_6/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_6/attention/self/transpose_2/permConst*
_output_shapes
:*%
valueB"             *
dtype0
?
/bert/encoder/layer_6/attention/self/transpose_2	Transpose-bert/encoder/layer_6/attention/self/Reshape_24bert/encoder/layer_6/attention/self/transpose_2/perm*
Tperm0*
T0*'
_output_shapes
:?@
?
,bert/encoder/layer_6/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_6/attention/self/Softmax/bert/encoder/layer_6/attention/self/transpose_2*
adj_y( *
T0*'
_output_shapes
:?@*
adj_x( 
?
4bert/encoder/layer_6/attention/self/transpose_3/permConst*
dtype0*
_output_shapes
:*%
valueB"             
?
/bert/encoder/layer_6/attention/self/transpose_3	Transpose,bert/encoder/layer_6/attention/self/MatMul_14bert/encoder/layer_6/attention/self/transpose_3/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_6/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_6/attention/self/Reshape_3Reshape/bert/encoder/layer_6/attention/self/transpose_33bert/encoder/layer_6/attention/self/Reshape_3/shape* 
_output_shapes
:
??*
T0*
Tshape0
?
Ubert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Tbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0
?
_bert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
seed2 
?
Sbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel
?
2bert/encoder/layer_6/attention/output/dense/kernel
VariableV2*
shared_name *E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
9bert/encoder/layer_6/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_6/attention/output/dense/kernelObert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
7bert/encoder/layer_6/attention/output/dense/kernel/readIdentity2bert/encoder/layer_6/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel* 
_output_shapes
:
??
?
Bbert/encoder/layer_6/attention/output/dense/bias/Initializer/zerosConst*
_output_shapes	
:?*C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
valueB?*    *
dtype0
?
0bert/encoder/layer_6/attention/output/dense/bias
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
7bert/encoder/layer_6/attention/output/dense/bias/AssignAssign0bert/encoder/layer_6/attention/output/dense/biasBbert/encoder/layer_6/attention/output/dense/bias/Initializer/zeros*C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
5bert/encoder/layer_6/attention/output/dense/bias/readIdentity0bert/encoder/layer_6/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
_output_shapes	
:?
?
2bert/encoder/layer_6/attention/output/dense/MatMulMatMul-bert/encoder/layer_6/attention/self/Reshape_37bert/encoder/layer_6/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
3bert/encoder/layer_6/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_6/attention/output/dense/MatMul5bert/encoder/layer_6/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
)bert/encoder/layer_6/attention/output/addAdd3bert/encoder/layer_6/attention/output/dense/BiasAdd5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
??*
T0
?
Fbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zerosConst*G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
4bert/encoder/layer_6/attention/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
	container *
shape:?
?
;bert/encoder/layer_6/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_6/attention/output/LayerNorm/betaFbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta
?
9bert/encoder/layer_6/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_6/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
_output_shapes	
:?
?
Fbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_6/attention/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma*
	container *
shape:?
?
<bert/encoder/layer_6/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_6/attention/output/LayerNorm/gammaFbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma
?
:bert/encoder/layer_6/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_6/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma*
_output_shapes	
:?
?
Nbert/encoder/layer_6/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
<bert/encoder/layer_6/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_6/attention/output/addNbert/encoder/layer_6/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
Dbert/encoder/layer_6/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_6/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
Ibert/encoder/layer_6/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_6/attention/output/addDbert/encoder/layer_6/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Rbert/encoder/layer_6/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
@bert/encoder/layer_6/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_6/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_6/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *̼?+*
dtype0
?
=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_6/attention/output/LayerNorm/moments/variance?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	?*
T0
?
?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_6/attention/output/LayerNorm/gamma/read* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_6/attention/output/add=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_6/attention/output/LayerNorm/moments/mean=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_6/attention/output/LayerNorm/beta/read?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
??*
T0
?
Qbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
valueB"      *
dtype0
?
Pbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
valueB
 *
ף<
?
[bert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/shape*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0
?
Obert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_6/intermediate/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
	container *
shape:
??
?
5bert/encoder/layer_6/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_6/intermediate/dense/kernelKbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
3bert/encoder/layer_6/intermediate/dense/kernel/readIdentity.bert/encoder/layer_6/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
valueB:?
?
Dbert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
valueB
 *    
?
>bert/encoder/layer_6/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros/Const*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*

index_type0
?
,bert/encoder/layer_6/intermediate/dense/bias
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
3bert/encoder/layer_6/intermediate/dense/bias/AssignAssign,bert/encoder/layer_6/intermediate/dense/bias>bert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias
?
1bert/encoder/layer_6/intermediate/dense/bias/readIdentity,bert/encoder/layer_6/intermediate/dense/bias*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
_output_shapes	
:?*
T0
?
.bert/encoder/layer_6/intermediate/dense/MatMulMatMul?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_6/intermediate/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
/bert/encoder/layer_6/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_6/intermediate/dense/MatMul1bert/encoder/layer_6/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
r
-bert/encoder/layer_6/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_6/intermediate/dense/PowPow/bert/encoder/layer_6/intermediate/dense/BiasAdd-bert/encoder/layer_6/intermediate/dense/Pow/y*
T0* 
_output_shapes
:
??
r
-bert/encoder/layer_6/intermediate/dense/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *'7=
?
+bert/encoder/layer_6/intermediate/dense/mulMul-bert/encoder/layer_6/intermediate/dense/mul/x+bert/encoder/layer_6/intermediate/dense/Pow* 
_output_shapes
:
??*
T0
?
+bert/encoder/layer_6/intermediate/dense/addAdd/bert/encoder/layer_6/intermediate/dense/BiasAdd+bert/encoder/layer_6/intermediate/dense/mul*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_6/intermediate/dense/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 **BL?
?
-bert/encoder/layer_6/intermediate/dense/mul_1Mul/bert/encoder/layer_6/intermediate/dense/mul_1/x+bert/encoder/layer_6/intermediate/dense/add*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_6/intermediate/dense/TanhTanh-bert/encoder/layer_6/intermediate/dense/mul_1*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_6/intermediate/dense/add_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_6/intermediate/dense/add_1Add/bert/encoder/layer_6/intermediate/dense/add_1/x,bert/encoder/layer_6/intermediate/dense/Tanh* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_6/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_6/intermediate/dense/mul_2Mul/bert/encoder/layer_6/intermediate/dense/mul_2/x-bert/encoder/layer_6/intermediate/dense/add_1*
T0* 
_output_shapes
:
??
?
-bert/encoder/layer_6/intermediate/dense/mul_3Mul/bert/encoder/layer_6/intermediate/dense/BiasAdd-bert/encoder/layer_6/intermediate/dense/mul_2* 
_output_shapes
:
??*
T0
?
Kbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Jbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Lbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/stddevConst*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
Ubert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/shape*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Ibert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel* 
_output_shapes
:
??
?
Ebert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/mean*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel* 
_output_shapes
:
??*
T0
?
(bert/encoder/layer_6/output/dense/kernel
VariableV2*
shared_name *;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
/bert/encoder/layer_6/output/dense/kernel/AssignAssign(bert/encoder/layer_6/output/dense/kernelEbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
-bert/encoder/layer_6/output/dense/kernel/readIdentity(bert/encoder/layer_6/output/dense/kernel* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel
?
8bert/encoder/layer_6/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias*
valueB?*    
?
&bert/encoder/layer_6/output/dense/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias*
	container 
?
-bert/encoder/layer_6/output/dense/bias/AssignAssign&bert/encoder/layer_6/output/dense/bias8bert/encoder/layer_6/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
+bert/encoder/layer_6/output/dense/bias/readIdentity&bert/encoder/layer_6/output/dense/bias*
_output_shapes	
:?*
T0*9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias
?
(bert/encoder/layer_6/output/dense/MatMulMatMul-bert/encoder/layer_6/intermediate/dense/mul_3-bert/encoder/layer_6/output/dense/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
)bert/encoder/layer_6/output/dense/BiasAddBiasAdd(bert/encoder/layer_6/output/dense/MatMul+bert/encoder/layer_6/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
bert/encoder/layer_6/output/addAdd)bert/encoder/layer_6/output/dense/BiasAdd?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
<bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
valueB?*    
?
*bert/encoder/layer_6/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
	container *
shape:?
?
1bert/encoder/layer_6/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_6/output/LayerNorm/beta<bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
/bert/encoder/layer_6/output/LayerNorm/beta/readIdentity*bert/encoder/layer_6/output/LayerNorm/beta*
_output_shapes	
:?*
T0*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta
?
<bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
+bert/encoder/layer_6/output/LayerNorm/gamma
VariableV2*
shared_name *>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
2bert/encoder/layer_6/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_6/output/LayerNorm/gamma<bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/ones*
_output_shapes	
:?*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
validate_shape(
?
0bert/encoder/layer_6/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_6/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
_output_shapes	
:?
?
Dbert/encoder/layer_6/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
2bert/encoder/layer_6/output/LayerNorm/moments/meanMeanbert/encoder/layer_6/output/addDbert/encoder/layer_6/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
:bert/encoder/layer_6/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_6/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_6/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_6/output/add:bert/encoder/layer_6/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
??*
T0
?
Hbert/encoder/layer_6/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
6bert/encoder/layer_6/output/LayerNorm/moments/varianceMean?bert/encoder/layer_6/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_6/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
z
5bert/encoder/layer_6/output/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼?+
?
3bert/encoder/layer_6/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_6/output/LayerNorm/moments/variance5bert/encoder/layer_6/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
5bert/encoder/layer_6/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_6/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
3bert/encoder/layer_6/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_6/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_6/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_6/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_6/output/add3bert/encoder/layer_6/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
5bert/encoder/layer_6/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_6/output/LayerNorm/moments/mean3bert/encoder/layer_6/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
3bert/encoder/layer_6/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_6/output/LayerNorm/beta/read5bert/encoder/layer_6/output/LayerNorm/batchnorm/mul_2* 
_output_shapes
:
??*
T0
?
5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_6/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_6/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Sbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Qbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel
?
0bert/encoder/layer_7/attention/self/query/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
7bert/encoder/layer_7/attention/self/query/kernel/AssignAssign0bert/encoder/layer_7/attention/self/query/kernelMbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel
?
5bert/encoder/layer_7/attention/self/query/kernel/readIdentity0bert/encoder/layer_7/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_7/attention/self/query/bias/Initializer/zerosConst*
_output_shapes	
:?*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
valueB?*    *
dtype0
?
.bert/encoder/layer_7/attention/self/query/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_7/attention/self/query/bias/AssignAssign.bert/encoder/layer_7/attention/self/query/bias@bert/encoder/layer_7/attention/self/query/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias
?
3bert/encoder/layer_7/attention/self/query/bias/readIdentity.bert/encoder/layer_7/attention/self/query/bias*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
_output_shapes	
:?*
T0
?
0bert/encoder/layer_7/attention/self/query/MatMulMatMul5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_15bert/encoder/layer_7/attention/self/query/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
1bert/encoder/layer_7/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_7/attention/self/query/MatMul3bert/encoder/layer_7/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Qbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
seed2 
?
Obert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel
?
Kbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_7/attention/self/key/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
	container 
?
5bert/encoder/layer_7/attention/self/key/kernel/AssignAssign.bert/encoder/layer_7/attention/self/key/kernelKbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
3bert/encoder/layer_7/attention/self/key/kernel/readIdentity.bert/encoder/layer_7/attention/self/key/kernel*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel* 
_output_shapes
:
??*
T0
?
>bert/encoder/layer_7/attention/self/key/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
,bert/encoder/layer_7/attention/self/key/bias
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
3bert/encoder/layer_7/attention/self/key/bias/AssignAssign,bert/encoder/layer_7/attention/self/key/bias>bert/encoder/layer_7/attention/self/key/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
validate_shape(
?
1bert/encoder/layer_7/attention/self/key/bias/readIdentity,bert/encoder/layer_7/attention/self/key/bias*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias
?
.bert/encoder/layer_7/attention/self/key/MatMulMatMul5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_13bert/encoder/layer_7/attention/self/key/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
/bert/encoder/layer_7/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_7/attention/self/key/MatMul1bert/encoder/layer_7/attention/self/key/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
Sbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
valueB"      *
dtype0
?
Rbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
valueB
 *
ף<
?
]bert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
seed2 
?
Qbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel
?
Mbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_7/attention/self/value/kernel
VariableV2*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
7bert/encoder/layer_7/attention/self/value/kernel/AssignAssign0bert/encoder/layer_7/attention/self/value/kernelMbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
validate_shape(
?
5bert/encoder/layer_7/attention/self/value/kernel/readIdentity0bert/encoder/layer_7/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_7/attention/self/value/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_7/attention/self/value/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias
?
5bert/encoder/layer_7/attention/self/value/bias/AssignAssign.bert/encoder/layer_7/attention/self/value/bias@bert/encoder/layer_7/attention/self/value/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
3bert/encoder/layer_7/attention/self/value/bias/readIdentity.bert/encoder/layer_7/attention/self/value/bias*
_output_shapes	
:?*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias
?
0bert/encoder/layer_7/attention/self/value/MatMulMatMul5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_15bert/encoder/layer_7/attention/self/value/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
1bert/encoder/layer_7/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_7/attention/self/value/MatMul3bert/encoder/layer_7/attention/self/value/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
?
1bert/encoder/layer_7/attention/self/Reshape/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
+bert/encoder/layer_7/attention/self/ReshapeReshape1bert/encoder/layer_7/attention/self/query/BiasAdd1bert/encoder/layer_7/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
2bert/encoder/layer_7/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_7/attention/self/transpose	Transpose+bert/encoder/layer_7/attention/self/Reshape2bert/encoder/layer_7/attention/self/transpose/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
3bert/encoder/layer_7/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_7/attention/self/Reshape_1Reshape/bert/encoder/layer_7/attention/self/key/BiasAdd3bert/encoder/layer_7/attention/self/Reshape_1/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
4bert/encoder/layer_7/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_7/attention/self/transpose_1	Transpose-bert/encoder/layer_7/attention/self/Reshape_14bert/encoder/layer_7/attention/self/transpose_1/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
*bert/encoder/layer_7/attention/self/MatMulBatchMatMul-bert/encoder/layer_7/attention/self/transpose/bert/encoder/layer_7/attention/self/transpose_1*
T0*(
_output_shapes
:??*
adj_x( *
adj_y(
n
)bert/encoder/layer_7/attention/self/Mul/yConst*
_output_shapes
: *
valueB
 *   >*
dtype0
?
'bert/encoder/layer_7/attention/self/MulMul*bert/encoder/layer_7/attention/self/MatMul)bert/encoder/layer_7/attention/self/Mul/y*
T0*(
_output_shapes
:??
|
2bert/encoder/layer_7/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
.bert/encoder/layer_7/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_7/attention/self/ExpandDims/dim*
T0*(
_output_shapes
:??*

Tdim0
n
)bert/encoder/layer_7/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_7/attention/self/subSub)bert/encoder/layer_7/attention/self/sub/x.bert/encoder/layer_7/attention/self/ExpandDims*(
_output_shapes
:??*
T0
p
+bert/encoder/layer_7/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
)bert/encoder/layer_7/attention/self/mul_1Mul'bert/encoder/layer_7/attention/self/sub+bert/encoder/layer_7/attention/self/mul_1/y*
T0*(
_output_shapes
:??
?
'bert/encoder/layer_7/attention/self/addAdd'bert/encoder/layer_7/attention/self/Mul)bert/encoder/layer_7/attention/self/mul_1*(
_output_shapes
:??*
T0
?
+bert/encoder/layer_7/attention/self/SoftmaxSoftmax'bert/encoder/layer_7/attention/self/add*
T0*(
_output_shapes
:??
?
3bert/encoder/layer_7/attention/self/Reshape_2/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_7/attention/self/Reshape_2Reshape1bert/encoder/layer_7/attention/self/value/BiasAdd3bert/encoder/layer_7/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_7/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_7/attention/self/transpose_2	Transpose-bert/encoder/layer_7/attention/self/Reshape_24bert/encoder/layer_7/attention/self/transpose_2/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
,bert/encoder/layer_7/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_7/attention/self/Softmax/bert/encoder/layer_7/attention/self/transpose_2*
adj_x( *
adj_y( *
T0*'
_output_shapes
:?@
?
4bert/encoder/layer_7/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_7/attention/self/transpose_3	Transpose,bert/encoder/layer_7/attention/self/MatMul_14bert/encoder/layer_7/attention/self/transpose_3/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
3bert/encoder/layer_7/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_7/attention/self/Reshape_3Reshape/bert/encoder/layer_7/attention/self/transpose_33bert/encoder/layer_7/attention/self/Reshape_3/shape*
Tshape0* 
_output_shapes
:
??*
T0
?
Ubert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Tbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
_bert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel
?
Sbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel* 
_output_shapes
:
??
?
2bert/encoder/layer_7/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
	container *
shape:
??
?
9bert/encoder/layer_7/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_7/attention/output/dense/kernelObert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
7bert/encoder/layer_7/attention/output/dense/kernel/readIdentity2bert/encoder/layer_7/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel* 
_output_shapes
:
??
?
Bbert/encoder/layer_7/attention/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
valueB?*    
?
0bert/encoder/layer_7/attention/output/dense/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias
?
7bert/encoder/layer_7/attention/output/dense/bias/AssignAssign0bert/encoder/layer_7/attention/output/dense/biasBbert/encoder/layer_7/attention/output/dense/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
5bert/encoder/layer_7/attention/output/dense/bias/readIdentity0bert/encoder/layer_7/attention/output/dense/bias*C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
_output_shapes	
:?*
T0
?
2bert/encoder/layer_7/attention/output/dense/MatMulMatMul-bert/encoder/layer_7/attention/self/Reshape_37bert/encoder/layer_7/attention/output/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
3bert/encoder/layer_7/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_7/attention/output/dense/MatMul5bert/encoder/layer_7/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
)bert/encoder/layer_7/attention/output/addAdd3bert/encoder/layer_7/attention/output/dense/BiasAdd5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
Fbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zerosConst*G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
4bert/encoder/layer_7/attention/output/LayerNorm/beta
VariableV2*
_output_shapes	
:?*
shared_name *G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
	container *
shape:?*
dtype0
?
;bert/encoder/layer_7/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_7/attention/output/LayerNorm/betaFbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
9bert/encoder/layer_7/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_7/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
_output_shapes	
:?
?
Fbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_7/attention/output/LayerNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma
?
<bert/encoder/layer_7/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_7/attention/output/LayerNorm/gammaFbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/ones*
T0*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
:bert/encoder/layer_7/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_7/attention/output/LayerNorm/gamma*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
_output_shapes	
:?*
T0
?
Nbert/encoder/layer_7/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
<bert/encoder/layer_7/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_7/attention/output/addNbert/encoder/layer_7/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
Dbert/encoder/layer_7/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_7/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
Ibert/encoder/layer_7/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_7/attention/output/addDbert/encoder/layer_7/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Rbert/encoder/layer_7/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
@bert/encoder/layer_7/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_7/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_7/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_7/attention/output/LayerNorm/moments/variance?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_7/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_7/attention/output/add=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_7/attention/output/LayerNorm/moments/mean=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_7/attention/output/LayerNorm/beta/read?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Qbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Obert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_7/intermediate/dense/kernel
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
5bert/encoder/layer_7/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_7/intermediate/dense/kernelKbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
3bert/encoder/layer_7/intermediate/dense/kernel/readIdentity.bert/encoder/layer_7/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
valueB:?*
dtype0*
_output_shapes
:
?
Dbert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros/ConstConst*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>bert/encoder/layer_7/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros/Const*
T0*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*

index_type0*
_output_shapes	
:?
?
,bert/encoder/layer_7/intermediate/dense/bias
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
3bert/encoder/layer_7/intermediate/dense/bias/AssignAssign,bert/encoder/layer_7/intermediate/dense/bias>bert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros*
T0*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
1bert/encoder/layer_7/intermediate/dense/bias/readIdentity,bert/encoder/layer_7/intermediate/dense/bias*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
_output_shapes	
:?*
T0
?
.bert/encoder/layer_7/intermediate/dense/MatMulMatMul?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_7/intermediate/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
/bert/encoder/layer_7/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_7/intermediate/dense/MatMul1bert/encoder/layer_7/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
r
-bert/encoder/layer_7/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_7/intermediate/dense/PowPow/bert/encoder/layer_7/intermediate/dense/BiasAdd-bert/encoder/layer_7/intermediate/dense/Pow/y*
T0* 
_output_shapes
:
??
r
-bert/encoder/layer_7/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_7/intermediate/dense/mulMul-bert/encoder/layer_7/intermediate/dense/mul/x+bert/encoder/layer_7/intermediate/dense/Pow*
T0* 
_output_shapes
:
??
?
+bert/encoder/layer_7/intermediate/dense/addAdd/bert/encoder/layer_7/intermediate/dense/BiasAdd+bert/encoder/layer_7/intermediate/dense/mul*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_7/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_7/intermediate/dense/mul_1Mul/bert/encoder/layer_7/intermediate/dense/mul_1/x+bert/encoder/layer_7/intermediate/dense/add* 
_output_shapes
:
??*
T0
?
,bert/encoder/layer_7/intermediate/dense/TanhTanh-bert/encoder/layer_7/intermediate/dense/mul_1*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_7/intermediate/dense/add_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_7/intermediate/dense/add_1Add/bert/encoder/layer_7/intermediate/dense/add_1/x,bert/encoder/layer_7/intermediate/dense/Tanh*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_7/intermediate/dense/mul_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
?
-bert/encoder/layer_7/intermediate/dense/mul_2Mul/bert/encoder/layer_7/intermediate/dense/mul_2/x-bert/encoder/layer_7/intermediate/dense/add_1* 
_output_shapes
:
??*
T0
?
-bert/encoder/layer_7/intermediate/dense/mul_3Mul/bert/encoder/layer_7/intermediate/dense/BiasAdd-bert/encoder/layer_7/intermediate/dense/mul_2*
T0* 
_output_shapes
:
??
?
Kbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Jbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Lbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/stddevConst*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
Ubert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/shape*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??
?
Ibert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel* 
_output_shapes
:
??
?
Ebert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel
?
(bert/encoder/layer_7/output/dense/kernel
VariableV2*
shared_name *;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
/bert/encoder/layer_7/output/dense/kernel/AssignAssign(bert/encoder/layer_7/output/dense/kernelEbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
-bert/encoder/layer_7/output/dense/kernel/readIdentity(bert/encoder/layer_7/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel* 
_output_shapes
:
??
?
8bert/encoder/layer_7/output/dense/bias/Initializer/zerosConst*9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
&bert/encoder/layer_7/output/dense/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias
?
-bert/encoder/layer_7/output/dense/bias/AssignAssign&bert/encoder/layer_7/output/dense/bias8bert/encoder/layer_7/output/dense/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias*
validate_shape(
?
+bert/encoder/layer_7/output/dense/bias/readIdentity&bert/encoder/layer_7/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias*
_output_shapes	
:?
?
(bert/encoder/layer_7/output/dense/MatMulMatMul-bert/encoder/layer_7/intermediate/dense/mul_3-bert/encoder/layer_7/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
)bert/encoder/layer_7/output/dense/BiasAddBiasAdd(bert/encoder/layer_7/output/dense/MatMul+bert/encoder/layer_7/output/dense/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
?
bert/encoder/layer_7/output/addAdd)bert/encoder/layer_7/output/dense/BiasAdd?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
<bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
*bert/encoder/layer_7/output/LayerNorm/beta
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
	container 
?
1bert/encoder/layer_7/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_7/output/LayerNorm/beta<bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zeros*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
/bert/encoder/layer_7/output/LayerNorm/beta/readIdentity*bert/encoder/layer_7/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
_output_shapes	
:?
?
<bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/onesConst*
_output_shapes	
:?*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
valueB?*  ??*
dtype0
?
+bert/encoder/layer_7/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
	container *
shape:?
?
2bert/encoder/layer_7/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_7/output/LayerNorm/gamma<bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
0bert/encoder/layer_7/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_7/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
_output_shapes	
:?
?
Dbert/encoder/layer_7/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
?
2bert/encoder/layer_7/output/LayerNorm/moments/meanMeanbert/encoder/layer_7/output/addDbert/encoder/layer_7/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
:bert/encoder/layer_7/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_7/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_7/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_7/output/add:bert/encoder/layer_7/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Hbert/encoder/layer_7/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
6bert/encoder/layer_7/output/LayerNorm/moments/varianceMean?bert/encoder/layer_7/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_7/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
z
5bert/encoder/layer_7/output/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼?+
?
3bert/encoder/layer_7/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_7/output/LayerNorm/moments/variance5bert/encoder/layer_7/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
5bert/encoder/layer_7/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_7/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
3bert/encoder/layer_7/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_7/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_7/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_7/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_7/output/add3bert/encoder/layer_7/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_7/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_7/output/LayerNorm/moments/mean3bert/encoder/layer_7/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
3bert/encoder/layer_7/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_7/output/LayerNorm/beta/read5bert/encoder/layer_7/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_7/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_7/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Sbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
valueB
 *
ף<
?
]bert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/shape*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:
??
?
Qbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel
?
Mbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel
?
0bert/encoder/layer_8/attention/self/query/kernel
VariableV2*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel
?
7bert/encoder/layer_8/attention/self/query/kernel/AssignAssign0bert/encoder/layer_8/attention/self/query/kernelMbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_8/attention/self/query/kernel/readIdentity0bert/encoder/layer_8/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_8/attention/self/query/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_8/attention/self/query/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_8/attention/self/query/bias/AssignAssign.bert/encoder/layer_8/attention/self/query/bias@bert/encoder/layer_8/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
3bert/encoder/layer_8/attention/self/query/bias/readIdentity.bert/encoder/layer_8/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
_output_shapes	
:?
?
0bert/encoder/layer_8/attention/self/query/MatMulMatMul5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_15bert/encoder/layer_8/attention/self/query/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
1bert/encoder/layer_8/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_8/attention/self/query/MatMul3bert/encoder/layer_8/attention/self/query/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
Qbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
seed2 
?
Obert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_8/attention/self/key/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
	container *
shape:
??
?
5bert/encoder/layer_8/attention/self/key/kernel/AssignAssign.bert/encoder/layer_8/attention/self/key/kernelKbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
3bert/encoder/layer_8/attention/self/key/kernel/readIdentity.bert/encoder/layer_8/attention/self/key/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel* 
_output_shapes
:
??
?
>bert/encoder/layer_8/attention/self/key/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
,bert/encoder/layer_8/attention/self/key/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
	container *
shape:?
?
3bert/encoder/layer_8/attention/self/key/bias/AssignAssign,bert/encoder/layer_8/attention/self/key/bias>bert/encoder/layer_8/attention/self/key/bias/Initializer/zeros*
T0*?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
1bert/encoder/layer_8/attention/self/key/bias/readIdentity,bert/encoder/layer_8/attention/self/key/bias*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias
?
.bert/encoder/layer_8/attention/self/key/MatMulMatMul5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_13bert/encoder/layer_8/attention/self/key/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
/bert/encoder/layer_8/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_8/attention/self/key/MatMul1bert/encoder/layer_8/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Sbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
seed2 
?
Qbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel
?
Mbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_8/attention/self/value/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
7bert/encoder/layer_8/attention/self/value/kernel/AssignAssign0bert/encoder/layer_8/attention/self/value/kernelMbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_8/attention/self/value/kernel/readIdentity0bert/encoder/layer_8/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_8/attention/self/value/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_8/attention/self/value/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias
?
5bert/encoder/layer_8/attention/self/value/bias/AssignAssign.bert/encoder/layer_8/attention/self/value/bias@bert/encoder/layer_8/attention/self/value/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
3bert/encoder/layer_8/attention/self/value/bias/readIdentity.bert/encoder/layer_8/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias*
_output_shapes	
:?
?
0bert/encoder/layer_8/attention/self/value/MatMulMatMul5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_15bert/encoder/layer_8/attention/self/value/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
1bert/encoder/layer_8/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_8/attention/self/value/MatMul3bert/encoder/layer_8/attention/self/value/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
?
1bert/encoder/layer_8/attention/self/Reshape/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
+bert/encoder/layer_8/attention/self/ReshapeReshape1bert/encoder/layer_8/attention/self/query/BiasAdd1bert/encoder/layer_8/attention/self/Reshape/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
2bert/encoder/layer_8/attention/self/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
?
-bert/encoder/layer_8/attention/self/transpose	Transpose+bert/encoder/layer_8/attention/self/Reshape2bert/encoder/layer_8/attention/self/transpose/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
3bert/encoder/layer_8/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_8/attention/self/Reshape_1Reshape/bert/encoder/layer_8/attention/self/key/BiasAdd3bert/encoder/layer_8/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_8/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_8/attention/self/transpose_1	Transpose-bert/encoder/layer_8/attention/self/Reshape_14bert/encoder/layer_8/attention/self/transpose_1/perm*
Tperm0*
T0*'
_output_shapes
:?@
?
*bert/encoder/layer_8/attention/self/MatMulBatchMatMul-bert/encoder/layer_8/attention/self/transpose/bert/encoder/layer_8/attention/self/transpose_1*
adj_y(*
T0*(
_output_shapes
:??*
adj_x( 
n
)bert/encoder/layer_8/attention/self/Mul/yConst*
dtype0*
_output_shapes
: *
valueB
 *   >
?
'bert/encoder/layer_8/attention/self/MulMul*bert/encoder/layer_8/attention/self/MatMul)bert/encoder/layer_8/attention/self/Mul/y*
T0*(
_output_shapes
:??
|
2bert/encoder/layer_8/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
.bert/encoder/layer_8/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_8/attention/self/ExpandDims/dim*(
_output_shapes
:??*

Tdim0*
T0
n
)bert/encoder/layer_8/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_8/attention/self/subSub)bert/encoder/layer_8/attention/self/sub/x.bert/encoder/layer_8/attention/self/ExpandDims*(
_output_shapes
:??*
T0
p
+bert/encoder/layer_8/attention/self/mul_1/yConst*
dtype0*
_output_shapes
: *
valueB
 * @?
?
)bert/encoder/layer_8/attention/self/mul_1Mul'bert/encoder/layer_8/attention/self/sub+bert/encoder/layer_8/attention/self/mul_1/y*
T0*(
_output_shapes
:??
?
'bert/encoder/layer_8/attention/self/addAdd'bert/encoder/layer_8/attention/self/Mul)bert/encoder/layer_8/attention/self/mul_1*(
_output_shapes
:??*
T0
?
+bert/encoder/layer_8/attention/self/SoftmaxSoftmax'bert/encoder/layer_8/attention/self/add*
T0*(
_output_shapes
:??
?
3bert/encoder/layer_8/attention/self/Reshape_2/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_8/attention/self/Reshape_2Reshape1bert/encoder/layer_8/attention/self/value/BiasAdd3bert/encoder/layer_8/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_8/attention/self/transpose_2/permConst*
_output_shapes
:*%
valueB"             *
dtype0
?
/bert/encoder/layer_8/attention/self/transpose_2	Transpose-bert/encoder/layer_8/attention/self/Reshape_24bert/encoder/layer_8/attention/self/transpose_2/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
,bert/encoder/layer_8/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_8/attention/self/Softmax/bert/encoder/layer_8/attention/self/transpose_2*
adj_x( *
adj_y( *
T0*'
_output_shapes
:?@
?
4bert/encoder/layer_8/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_8/attention/self/transpose_3	Transpose,bert/encoder/layer_8/attention/self/MatMul_14bert/encoder/layer_8/attention/self/transpose_3/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
3bert/encoder/layer_8/attention/self/Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"?      
?
-bert/encoder/layer_8/attention/self/Reshape_3Reshape/bert/encoder/layer_8/attention/self/transpose_33bert/encoder/layer_8/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:
??
?
Ubert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Tbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
valueB
 *    
?
Vbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0
?
_bert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/shape*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Sbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel* 
_output_shapes
:
??
?
2bert/encoder/layer_8/attention/output/dense/kernel
VariableV2*
shared_name *E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
9bert/encoder/layer_8/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_8/attention/output/dense/kernelObert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
7bert/encoder/layer_8/attention/output/dense/kernel/readIdentity2bert/encoder/layer_8/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel* 
_output_shapes
:
??
?
Bbert/encoder/layer_8/attention/output/dense/bias/Initializer/zerosConst*C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
0bert/encoder/layer_8/attention/output/dense/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
	container 
?
7bert/encoder/layer_8/attention/output/dense/bias/AssignAssign0bert/encoder/layer_8/attention/output/dense/biasBbert/encoder/layer_8/attention/output/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias
?
5bert/encoder/layer_8/attention/output/dense/bias/readIdentity0bert/encoder/layer_8/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
_output_shapes	
:?
?
2bert/encoder/layer_8/attention/output/dense/MatMulMatMul-bert/encoder/layer_8/attention/self/Reshape_37bert/encoder/layer_8/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
3bert/encoder/layer_8/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_8/attention/output/dense/MatMul5bert/encoder/layer_8/attention/output/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
)bert/encoder/layer_8/attention/output/addAdd3bert/encoder/layer_8/attention/output/dense/BiasAdd5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
??*
T0
?
Fbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
valueB?*    
?
4bert/encoder/layer_8/attention/output/LayerNorm/beta
VariableV2*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name 
?
;bert/encoder/layer_8/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_8/attention/output/LayerNorm/betaFbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
9bert/encoder/layer_8/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_8/attention/output/LayerNorm/beta*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
_output_shapes	
:?*
T0
?
Fbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
valueB?*  ??
?
5bert/encoder/layer_8/attention/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
	container *
shape:?
?
<bert/encoder/layer_8/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_8/attention/output/LayerNorm/gammaFbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/ones*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
validate_shape(
?
:bert/encoder/layer_8/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_8/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
_output_shapes	
:?
?
Nbert/encoder/layer_8/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
?
<bert/encoder/layer_8/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_8/attention/output/addNbert/encoder/layer_8/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
Dbert/encoder/layer_8/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_8/attention/output/LayerNorm/moments/mean*
_output_shapes
:	?*
T0
?
Ibert/encoder/layer_8/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_8/attention/output/addDbert/encoder/layer_8/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Rbert/encoder/layer_8/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
@bert/encoder/layer_8/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_8/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_8/attention/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼?+
?
=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_8/attention/output/LayerNorm/moments/variance?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add*
_output_shapes
:	?*
T0
?
=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_8/attention/output/LayerNorm/gamma/read* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_8/attention/output/add=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_8/attention/output/LayerNorm/moments/mean=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_8/attention/output/LayerNorm/beta/read?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul_2* 
_output_shapes
:
??*
T0
?
?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
??*
T0
?
Qbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
valueB
 *    *
dtype0
?
Rbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
valueB
 *
ף<
?
[bert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel
?
Obert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel* 
_output_shapes
:
??
?
.bert/encoder/layer_8/intermediate/dense/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
	container 
?
5bert/encoder/layer_8/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_8/intermediate/dense/kernelKbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel
?
3bert/encoder/layer_8/intermediate/dense/kernel/readIdentity.bert/encoder/layer_8/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
valueB:?*
dtype0*
_output_shapes
:
?
Dbert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros/ConstConst*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
>bert/encoder/layer_8/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros/Const*
T0*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*

index_type0*
_output_shapes	
:?
?
,bert/encoder/layer_8/intermediate/dense/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
	container 
?
3bert/encoder/layer_8/intermediate/dense/bias/AssignAssign,bert/encoder/layer_8/intermediate/dense/bias>bert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_8/intermediate/dense/bias/readIdentity,bert/encoder/layer_8/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
_output_shapes	
:?
?
.bert/encoder/layer_8/intermediate/dense/MatMulMatMul?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_8/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_8/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_8/intermediate/dense/MatMul1bert/encoder/layer_8/intermediate/dense/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
r
-bert/encoder/layer_8/intermediate/dense/Pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *  @@
?
+bert/encoder/layer_8/intermediate/dense/PowPow/bert/encoder/layer_8/intermediate/dense/BiasAdd-bert/encoder/layer_8/intermediate/dense/Pow/y* 
_output_shapes
:
??*
T0
r
-bert/encoder/layer_8/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_8/intermediate/dense/mulMul-bert/encoder/layer_8/intermediate/dense/mul/x+bert/encoder/layer_8/intermediate/dense/Pow* 
_output_shapes
:
??*
T0
?
+bert/encoder/layer_8/intermediate/dense/addAdd/bert/encoder/layer_8/intermediate/dense/BiasAdd+bert/encoder/layer_8/intermediate/dense/mul* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_8/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_8/intermediate/dense/mul_1Mul/bert/encoder/layer_8/intermediate/dense/mul_1/x+bert/encoder/layer_8/intermediate/dense/add*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_8/intermediate/dense/TanhTanh-bert/encoder/layer_8/intermediate/dense/mul_1* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_8/intermediate/dense/add_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_8/intermediate/dense/add_1Add/bert/encoder/layer_8/intermediate/dense/add_1/x,bert/encoder/layer_8/intermediate/dense/Tanh* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_8/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_8/intermediate/dense/mul_2Mul/bert/encoder/layer_8/intermediate/dense/mul_2/x-bert/encoder/layer_8/intermediate/dense/add_1*
T0* 
_output_shapes
:
??
?
-bert/encoder/layer_8/intermediate/dense/mul_3Mul/bert/encoder/layer_8/intermediate/dense/BiasAdd-bert/encoder/layer_8/intermediate/dense/mul_2* 
_output_shapes
:
??*
T0
?
Kbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Jbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Lbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
valueB
 *
ף<*
dtype0
?
Ubert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
seed2 
?
Ibert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel* 
_output_shapes
:
??
?
Ebert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel* 
_output_shapes
:
??
?
(bert/encoder/layer_8/output/dense/kernel
VariableV2* 
_output_shapes
:
??*
shared_name *;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
	container *
shape:
??*
dtype0
?
/bert/encoder/layer_8/output/dense/kernel/AssignAssign(bert/encoder/layer_8/output/dense/kernelEbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
-bert/encoder/layer_8/output/dense/kernel/readIdentity(bert/encoder/layer_8/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel* 
_output_shapes
:
??
?
8bert/encoder/layer_8/output/dense/bias/Initializer/zerosConst*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
&bert/encoder/layer_8/output/dense/bias
VariableV2*
shared_name *9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
-bert/encoder/layer_8/output/dense/bias/AssignAssign&bert/encoder/layer_8/output/dense/bias8bert/encoder/layer_8/output/dense/bias/Initializer/zeros*
T0*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
+bert/encoder/layer_8/output/dense/bias/readIdentity&bert/encoder/layer_8/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
_output_shapes	
:?
?
(bert/encoder/layer_8/output/dense/MatMulMatMul-bert/encoder/layer_8/intermediate/dense/mul_3-bert/encoder/layer_8/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
)bert/encoder/layer_8/output/dense/BiasAddBiasAdd(bert/encoder/layer_8/output/dense/MatMul+bert/encoder/layer_8/output/dense/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
?
bert/encoder/layer_8/output/addAdd)bert/encoder/layer_8/output/dense/BiasAdd?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
<bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
valueB?*    
?
*bert/encoder/layer_8/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
	container *
shape:?
?
1bert/encoder/layer_8/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_8/output/LayerNorm/beta<bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
/bert/encoder/layer_8/output/LayerNorm/beta/readIdentity*bert/encoder/layer_8/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
_output_shapes	
:?
?
<bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
+bert/encoder/layer_8/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
	container *
shape:?
?
2bert/encoder/layer_8/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_8/output/LayerNorm/gamma<bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/ones*
T0*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
0bert/encoder/layer_8/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_8/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
_output_shapes	
:?
?
Dbert/encoder/layer_8/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
2bert/encoder/layer_8/output/LayerNorm/moments/meanMeanbert/encoder/layer_8/output/addDbert/encoder/layer_8/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
:bert/encoder/layer_8/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_8/output/LayerNorm/moments/mean*
_output_shapes
:	?*
T0
?
?bert/encoder/layer_8/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_8/output/add:bert/encoder/layer_8/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
??*
T0
?
Hbert/encoder/layer_8/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
6bert/encoder/layer_8/output/LayerNorm/moments/varianceMean?bert/encoder/layer_8/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_8/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
z
5bert/encoder/layer_8/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
3bert/encoder/layer_8/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_8/output/LayerNorm/moments/variance5bert/encoder/layer_8/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
5bert/encoder/layer_8/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_8/output/LayerNorm/batchnorm/add*
_output_shapes
:	?*
T0
?
3bert/encoder/layer_8/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_8/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_8/output/LayerNorm/gamma/read* 
_output_shapes
:
??*
T0
?
5bert/encoder/layer_8/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_8/output/add3bert/encoder/layer_8/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_8/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_8/output/LayerNorm/moments/mean3bert/encoder/layer_8/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
3bert/encoder/layer_8/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_8/output/LayerNorm/beta/read5bert/encoder/layer_8/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_8/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_8/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Sbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Qbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel
?
Mbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel* 
_output_shapes
:
??
?
0bert/encoder/layer_9/attention/self/query/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
7bert/encoder/layer_9/attention/self/query/kernel/AssignAssign0bert/encoder/layer_9/attention/self/query/kernelMbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_9/attention/self/query/kernel/readIdentity0bert/encoder/layer_9/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_9/attention/self/query/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_9/attention/self/query/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_9/attention/self/query/bias/AssignAssign.bert/encoder/layer_9/attention/self/query/bias@bert/encoder/layer_9/attention/self/query/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
validate_shape(
?
3bert/encoder/layer_9/attention/self/query/bias/readIdentity.bert/encoder/layer_9/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
_output_shapes	
:?
?
0bert/encoder/layer_9/attention/self/query/MatMulMatMul5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_15bert/encoder/layer_9/attention/self/query/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
1bert/encoder/layer_9/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_9/attention/self/query/MatMul3bert/encoder/layer_9/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Qbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
valueB
 *    *
dtype0
?
Rbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel
?
Obert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel
?
Kbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel
?
.bert/encoder/layer_9/attention/self/key/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
	container 
?
5bert/encoder/layer_9/attention/self/key/kernel/AssignAssign.bert/encoder/layer_9/attention/self/key/kernelKbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
3bert/encoder/layer_9/attention/self/key/kernel/readIdentity.bert/encoder/layer_9/attention/self/key/kernel*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel* 
_output_shapes
:
??*
T0
?
>bert/encoder/layer_9/attention/self/key/bias/Initializer/zerosConst*
_output_shapes	
:?*?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
valueB?*    *
dtype0
?
,bert/encoder/layer_9/attention/self/key/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
	container *
shape:?
?
3bert/encoder/layer_9/attention/self/key/bias/AssignAssign,bert/encoder/layer_9/attention/self/key/bias>bert/encoder/layer_9/attention/self/key/bias/Initializer/zeros*
T0*?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
1bert/encoder/layer_9/attention/self/key/bias/readIdentity,bert/encoder/layer_9/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
_output_shapes	
:?
?
.bert/encoder/layer_9/attention/self/key/MatMulMatMul5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_13bert/encoder/layer_9/attention/self/key/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_9/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_9/attention/self/key/MatMul1bert/encoder/layer_9/attention/self/key/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
Sbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Rbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
]bert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
??*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
seed2 *
dtype0
?
Qbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel* 
_output_shapes
:
??
?
Mbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel
?
0bert/encoder/layer_9/attention/self/value/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
7bert/encoder/layer_9/attention/self/value/kernel/AssignAssign0bert/encoder/layer_9/attention/self/value/kernelMbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
5bert/encoder/layer_9/attention/self/value/kernel/readIdentity0bert/encoder/layer_9/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel* 
_output_shapes
:
??
?
@bert/encoder/layer_9/attention/self/value/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_9/attention/self/value/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_9/attention/self/value/bias/AssignAssign.bert/encoder/layer_9/attention/self/value/bias@bert/encoder/layer_9/attention/self/value/bias/Initializer/zeros*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
3bert/encoder/layer_9/attention/self/value/bias/readIdentity.bert/encoder/layer_9/attention/self/value/bias*
_output_shapes	
:?*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias
?
0bert/encoder/layer_9/attention/self/value/MatMulMatMul5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_15bert/encoder/layer_9/attention/self/value/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
1bert/encoder/layer_9/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_9/attention/self/value/MatMul3bert/encoder/layer_9/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
1bert/encoder/layer_9/attention/self/Reshape/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
+bert/encoder/layer_9/attention/self/ReshapeReshape1bert/encoder/layer_9/attention/self/query/BiasAdd1bert/encoder/layer_9/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
2bert/encoder/layer_9/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_9/attention/self/transpose	Transpose+bert/encoder/layer_9/attention/self/Reshape2bert/encoder/layer_9/attention/self/transpose/perm*
Tperm0*
T0*'
_output_shapes
:?@
?
3bert/encoder/layer_9/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_9/attention/self/Reshape_1Reshape/bert/encoder/layer_9/attention/self/key/BiasAdd3bert/encoder/layer_9/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
4bert/encoder/layer_9/attention/self/transpose_1/permConst*
_output_shapes
:*%
valueB"             *
dtype0
?
/bert/encoder/layer_9/attention/self/transpose_1	Transpose-bert/encoder/layer_9/attention/self/Reshape_14bert/encoder/layer_9/attention/self/transpose_1/perm*
Tperm0*
T0*'
_output_shapes
:?@
?
*bert/encoder/layer_9/attention/self/MatMulBatchMatMul-bert/encoder/layer_9/attention/self/transpose/bert/encoder/layer_9/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:??
n
)bert/encoder/layer_9/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_9/attention/self/MulMul*bert/encoder/layer_9/attention/self/MatMul)bert/encoder/layer_9/attention/self/Mul/y*(
_output_shapes
:??*
T0
|
2bert/encoder/layer_9/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
.bert/encoder/layer_9/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_9/attention/self/ExpandDims/dim*(
_output_shapes
:??*

Tdim0*
T0
n
)bert/encoder/layer_9/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
'bert/encoder/layer_9/attention/self/subSub)bert/encoder/layer_9/attention/self/sub/x.bert/encoder/layer_9/attention/self/ExpandDims*
T0*(
_output_shapes
:??
p
+bert/encoder/layer_9/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
)bert/encoder/layer_9/attention/self/mul_1Mul'bert/encoder/layer_9/attention/self/sub+bert/encoder/layer_9/attention/self/mul_1/y*
T0*(
_output_shapes
:??
?
'bert/encoder/layer_9/attention/self/addAdd'bert/encoder/layer_9/attention/self/Mul)bert/encoder/layer_9/attention/self/mul_1*(
_output_shapes
:??*
T0
?
+bert/encoder/layer_9/attention/self/SoftmaxSoftmax'bert/encoder/layer_9/attention/self/add*
T0*(
_output_shapes
:??
?
3bert/encoder/layer_9/attention/self/Reshape_2/shapeConst*
_output_shapes
:*%
valueB"   ?      @   *
dtype0
?
-bert/encoder/layer_9/attention/self/Reshape_2Reshape1bert/encoder/layer_9/attention/self/value/BiasAdd3bert/encoder/layer_9/attention/self/Reshape_2/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
4bert/encoder/layer_9/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_9/attention/self/transpose_2	Transpose-bert/encoder/layer_9/attention/self/Reshape_24bert/encoder/layer_9/attention/self/transpose_2/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
,bert/encoder/layer_9/attention/self/MatMul_1BatchMatMul+bert/encoder/layer_9/attention/self/Softmax/bert/encoder/layer_9/attention/self/transpose_2*
adj_x( *
adj_y( *
T0*'
_output_shapes
:?@
?
4bert/encoder/layer_9/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
/bert/encoder/layer_9/attention/self/transpose_3	Transpose,bert/encoder/layer_9/attention/self/MatMul_14bert/encoder/layer_9/attention/self/transpose_3/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
3bert/encoder/layer_9/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
-bert/encoder/layer_9/attention/self/Reshape_3Reshape/bert/encoder/layer_9/attention/self/transpose_33bert/encoder/layer_9/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:
??
?
Ubert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Tbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
_bert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
??*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
seed2 *
dtype0
?
Sbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel
?
2bert/encoder/layer_9/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
	container *
shape:
??
?
9bert/encoder/layer_9/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_9/attention/output/dense/kernelObert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
7bert/encoder/layer_9/attention/output/dense/kernel/readIdentity2bert/encoder/layer_9/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel* 
_output_shapes
:
??
?
Bbert/encoder/layer_9/attention/output/dense/bias/Initializer/zerosConst*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
0bert/encoder/layer_9/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
	container *
shape:?
?
7bert/encoder/layer_9/attention/output/dense/bias/AssignAssign0bert/encoder/layer_9/attention/output/dense/biasBbert/encoder/layer_9/attention/output/dense/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
validate_shape(
?
5bert/encoder/layer_9/attention/output/dense/bias/readIdentity0bert/encoder/layer_9/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
_output_shapes	
:?
?
2bert/encoder/layer_9/attention/output/dense/MatMulMatMul-bert/encoder/layer_9/attention/self/Reshape_37bert/encoder/layer_9/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
3bert/encoder/layer_9/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_9/attention/output/dense/MatMul5bert/encoder/layer_9/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
)bert/encoder/layer_9/attention/output/addAdd3bert/encoder/layer_9/attention/output/dense/BiasAdd5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
??*
T0
?
Fbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zerosConst*
_output_shapes	
:?*G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta*
valueB?*    *
dtype0
?
4bert/encoder/layer_9/attention/output/LayerNorm/beta
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta*
	container 
?
;bert/encoder/layer_9/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_9/attention/output/LayerNorm/betaFbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zeros*G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
9bert/encoder/layer_9/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_9/attention/output/LayerNorm/beta*
_output_shapes	
:?*
T0*G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta
?
Fbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_9/attention/output/LayerNorm/gamma
VariableV2*
shared_name *H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
<bert/encoder/layer_9/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_9/attention/output/LayerNorm/gammaFbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
:bert/encoder/layer_9/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_9/attention/output/LayerNorm/gamma*
_output_shapes	
:?*
T0*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma
?
Nbert/encoder/layer_9/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
<bert/encoder/layer_9/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_9/attention/output/addNbert/encoder/layer_9/attention/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
Dbert/encoder/layer_9/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_9/attention/output/LayerNorm/moments/mean*
_output_shapes
:	?*
T0
?
Ibert/encoder/layer_9/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_9/attention/output/addDbert/encoder/layer_9/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Rbert/encoder/layer_9/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0
?
@bert/encoder/layer_9/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_9/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_9/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
?
?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_9/attention/output/LayerNorm/moments/variance?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_9/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_9/attention/output/add=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_9/attention/output/LayerNorm/moments/mean=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_9/attention/output/LayerNorm/beta/read?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Qbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Pbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
[bert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
??*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
seed2 *
dtype0
?
Obert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel* 
_output_shapes
:
??
?
Kbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel
?
.bert/encoder/layer_9/intermediate/dense/kernel
VariableV2*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
5bert/encoder/layer_9/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_9/intermediate/dense/kernelKbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel
?
3bert/encoder/layer_9/intermediate/dense/kernel/readIdentity.bert/encoder/layer_9/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
valueB:?*
dtype0*
_output_shapes
:
?
Dbert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros/ConstConst*
_output_shapes
: *?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
valueB
 *    *
dtype0
?
>bert/encoder/layer_9/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros/Const*
T0*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*

index_type0*
_output_shapes	
:?
?
,bert/encoder/layer_9/intermediate/dense/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
	container 
?
3bert/encoder/layer_9/intermediate/dense/bias/AssignAssign,bert/encoder/layer_9/intermediate/dense/bias>bert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
1bert/encoder/layer_9/intermediate/dense/bias/readIdentity,bert/encoder/layer_9/intermediate/dense/bias*
_output_shapes	
:?*
T0*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias
?
.bert/encoder/layer_9/intermediate/dense/MatMulMatMul?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_9/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
/bert/encoder/layer_9/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_9/intermediate/dense/MatMul1bert/encoder/layer_9/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
r
-bert/encoder/layer_9/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_9/intermediate/dense/PowPow/bert/encoder/layer_9/intermediate/dense/BiasAdd-bert/encoder/layer_9/intermediate/dense/Pow/y*
T0* 
_output_shapes
:
??
r
-bert/encoder/layer_9/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
?
+bert/encoder/layer_9/intermediate/dense/mulMul-bert/encoder/layer_9/intermediate/dense/mul/x+bert/encoder/layer_9/intermediate/dense/Pow* 
_output_shapes
:
??*
T0
?
+bert/encoder/layer_9/intermediate/dense/addAdd/bert/encoder/layer_9/intermediate/dense/BiasAdd+bert/encoder/layer_9/intermediate/dense/mul* 
_output_shapes
:
??*
T0
t
/bert/encoder/layer_9/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_9/intermediate/dense/mul_1Mul/bert/encoder/layer_9/intermediate/dense/mul_1/x+bert/encoder/layer_9/intermediate/dense/add*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_9/intermediate/dense/TanhTanh-bert/encoder/layer_9/intermediate/dense/mul_1*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_9/intermediate/dense/add_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-bert/encoder/layer_9/intermediate/dense/add_1Add/bert/encoder/layer_9/intermediate/dense/add_1/x,bert/encoder/layer_9/intermediate/dense/Tanh*
T0* 
_output_shapes
:
??
t
/bert/encoder/layer_9/intermediate/dense/mul_2/xConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
?
-bert/encoder/layer_9/intermediate/dense/mul_2Mul/bert/encoder/layer_9/intermediate/dense/mul_2/x-bert/encoder/layer_9/intermediate/dense/add_1*
T0* 
_output_shapes
:
??
?
-bert/encoder/layer_9/intermediate/dense/mul_3Mul/bert/encoder/layer_9/intermediate/dense/BiasAdd-bert/encoder/layer_9/intermediate/dense/mul_2*
T0* 
_output_shapes
:
??
?
Kbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Jbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Lbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/stddevConst*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
Ubert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
seed2 
?
Ibert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel
?
Ebert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/mean*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel* 
_output_shapes
:
??*
T0
?
(bert/encoder/layer_9/output/dense/kernel
VariableV2*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel
?
/bert/encoder/layer_9/output/dense/kernel/AssignAssign(bert/encoder/layer_9/output/dense/kernelEbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
-bert/encoder/layer_9/output/dense/kernel/readIdentity(bert/encoder/layer_9/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel* 
_output_shapes
:
??
?
8bert/encoder/layer_9/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
valueB?*    
?
&bert/encoder/layer_9/output/dense/bias
VariableV2*
shared_name *9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
-bert/encoder/layer_9/output/dense/bias/AssignAssign&bert/encoder/layer_9/output/dense/bias8bert/encoder/layer_9/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
+bert/encoder/layer_9/output/dense/bias/readIdentity&bert/encoder/layer_9/output/dense/bias*
_output_shapes	
:?*
T0*9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias
?
(bert/encoder/layer_9/output/dense/MatMulMatMul-bert/encoder/layer_9/intermediate/dense/mul_3-bert/encoder/layer_9/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
)bert/encoder/layer_9/output/dense/BiasAddBiasAdd(bert/encoder/layer_9/output/dense/MatMul+bert/encoder/layer_9/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
bert/encoder/layer_9/output/addAdd)bert/encoder/layer_9/output/dense/BiasAdd?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
??*
T0
?
<bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
*bert/encoder/layer_9/output/LayerNorm/beta
VariableV2*
shared_name *=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
1bert/encoder/layer_9/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_9/output/LayerNorm/beta<bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
/bert/encoder/layer_9/output/LayerNorm/beta/readIdentity*bert/encoder/layer_9/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
_output_shapes	
:?
?
<bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
+bert/encoder/layer_9/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
	container *
shape:?
?
2bert/encoder/layer_9/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_9/output/LayerNorm/gamma<bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/ones*
_output_shapes	
:?*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
validate_shape(
?
0bert/encoder/layer_9/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_9/output/LayerNorm/gamma*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
_output_shapes	
:?*
T0
?
Dbert/encoder/layer_9/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
2bert/encoder/layer_9/output/LayerNorm/moments/meanMeanbert/encoder/layer_9/output/addDbert/encoder/layer_9/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
:bert/encoder/layer_9/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_9/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
?bert/encoder/layer_9/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_9/output/add:bert/encoder/layer_9/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Hbert/encoder/layer_9/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
6bert/encoder/layer_9/output/LayerNorm/moments/varianceMean?bert/encoder/layer_9/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_9/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
z
5bert/encoder/layer_9/output/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *̼?+*
dtype0
?
3bert/encoder/layer_9/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_9/output/LayerNorm/moments/variance5bert/encoder/layer_9/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
5bert/encoder/layer_9/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_9/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
3bert/encoder/layer_9/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_9/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_9/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_9/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_9/output/add3bert/encoder/layer_9/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
5bert/encoder/layer_9/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_9/output/LayerNorm/moments/mean3bert/encoder/layer_9/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
3bert/encoder/layer_9/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_9/output/LayerNorm/beta/read5bert/encoder/layer_9/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_9/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_9/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
??*
T0
?
Tbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Sbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/meanConst*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ubert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
valueB
 *
ף<*
dtype0
?
^bert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
??*

seed *
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
seed2 *
dtype0
?
Rbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/mulMul^bert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalUbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/stddev*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel* 
_output_shapes
:
??*
T0
?
Nbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normalAddRbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/mulSbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel* 
_output_shapes
:
??
?
1bert/encoder/layer_10/attention/self/query/kernel
VariableV2*
shared_name *D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
8bert/encoder/layer_10/attention/self/query/kernel/AssignAssign1bert/encoder/layer_10/attention/self/query/kernelNbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
6bert/encoder/layer_10/attention/self/query/kernel/readIdentity1bert/encoder/layer_10/attention/self/query/kernel*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel* 
_output_shapes
:
??
?
Abert/encoder/layer_10/attention/self/query/bias/Initializer/zerosConst*B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
/bert/encoder/layer_10/attention/self/query/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
	container *
shape:?
?
6bert/encoder/layer_10/attention/self/query/bias/AssignAssign/bert/encoder/layer_10/attention/self/query/biasAbert/encoder/layer_10/attention/self/query/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias
?
4bert/encoder/layer_10/attention/self/query/bias/readIdentity/bert/encoder/layer_10/attention/self/query/bias*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
_output_shapes	
:?
?
1bert/encoder/layer_10/attention/self/query/MatMulMatMul5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_16bert/encoder/layer_10/attention/self/query/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
2bert/encoder/layer_10/attention/self/query/BiasAddBiasAdd1bert/encoder/layer_10/attention/self/query/MatMul4bert/encoder/layer_10/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Rbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Qbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/meanConst*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Sbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
valueB
 *
ף<*
dtype0
?
\bert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
seed2 
?
Pbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/mulMul\bert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalSbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel
?
Lbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normalAddPbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/mulQbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/mean*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel* 
_output_shapes
:
??*
T0
?
/bert/encoder/layer_10/attention/self/key/kernel
VariableV2*
shared_name *B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??
?
6bert/encoder/layer_10/attention/self/key/kernel/AssignAssign/bert/encoder/layer_10/attention/self/key/kernelLbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
4bert/encoder/layer_10/attention/self/key/kernel/readIdentity/bert/encoder/layer_10/attention/self/key/kernel*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel* 
_output_shapes
:
??
?
?bert/encoder/layer_10/attention/self/key/bias/Initializer/zerosConst*@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
-bert/encoder/layer_10/attention/self/key/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
	container *
shape:?
?
4bert/encoder/layer_10/attention/self/key/bias/AssignAssign-bert/encoder/layer_10/attention/self/key/bias?bert/encoder/layer_10/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
2bert/encoder/layer_10/attention/self/key/bias/readIdentity-bert/encoder/layer_10/attention/self/key/bias*@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
_output_shapes	
:?*
T0
?
/bert/encoder/layer_10/attention/self/key/MatMulMatMul5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_14bert/encoder/layer_10/attention/self/key/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
0bert/encoder/layer_10/attention/self/key/BiasAddBiasAdd/bert/encoder/layer_10/attention/self/key/MatMul2bert/encoder/layer_10/attention/self/key/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
?
Tbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Sbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
valueB
 *    
?
Ubert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
^bert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel
?
Rbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/mulMul^bert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalUbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normalAddRbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/mulSbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel
?
1bert/encoder/layer_10/attention/self/value/kernel
VariableV2*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
8bert/encoder/layer_10/attention/self/value/kernel/AssignAssign1bert/encoder/layer_10/attention/self/value/kernelNbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel
?
6bert/encoder/layer_10/attention/self/value/kernel/readIdentity1bert/encoder/layer_10/attention/self/value/kernel* 
_output_shapes
:
??*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel
?
Abert/encoder/layer_10/attention/self/value/bias/Initializer/zerosConst*B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
/bert/encoder/layer_10/attention/self/value/bias
VariableV2*
shared_name *B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
6bert/encoder/layer_10/attention/self/value/bias/AssignAssign/bert/encoder/layer_10/attention/self/value/biasAbert/encoder/layer_10/attention/self/value/bias/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
4bert/encoder/layer_10/attention/self/value/bias/readIdentity/bert/encoder/layer_10/attention/self/value/bias*
_output_shapes	
:?*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias
?
1bert/encoder/layer_10/attention/self/value/MatMulMatMul5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_16bert/encoder/layer_10/attention/self/value/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
2bert/encoder/layer_10/attention/self/value/BiasAddBiasAdd1bert/encoder/layer_10/attention/self/value/MatMul4bert/encoder/layer_10/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
2bert/encoder/layer_10/attention/self/Reshape/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
,bert/encoder/layer_10/attention/self/ReshapeReshape2bert/encoder/layer_10/attention/self/query/BiasAdd2bert/encoder/layer_10/attention/self/Reshape/shape*'
_output_shapes
:?@*
T0*
Tshape0
?
3bert/encoder/layer_10/attention/self/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
?
.bert/encoder/layer_10/attention/self/transpose	Transpose,bert/encoder/layer_10/attention/self/Reshape3bert/encoder/layer_10/attention/self/transpose/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
4bert/encoder/layer_10/attention/self/Reshape_1/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
.bert/encoder/layer_10/attention/self/Reshape_1Reshape0bert/encoder/layer_10/attention/self/key/BiasAdd4bert/encoder/layer_10/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
5bert/encoder/layer_10/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
0bert/encoder/layer_10/attention/self/transpose_1	Transpose.bert/encoder/layer_10/attention/self/Reshape_15bert/encoder/layer_10/attention/self/transpose_1/perm*
Tperm0*
T0*'
_output_shapes
:?@
?
+bert/encoder/layer_10/attention/self/MatMulBatchMatMul.bert/encoder/layer_10/attention/self/transpose0bert/encoder/layer_10/attention/self/transpose_1*(
_output_shapes
:??*
adj_x( *
adj_y(*
T0
o
*bert/encoder/layer_10/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
(bert/encoder/layer_10/attention/self/MulMul+bert/encoder/layer_10/attention/self/MatMul*bert/encoder/layer_10/attention/self/Mul/y*
T0*(
_output_shapes
:??
}
3bert/encoder/layer_10/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
/bert/encoder/layer_10/attention/self/ExpandDims
ExpandDimsbert/encoder/mul3bert/encoder/layer_10/attention/self/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:??
o
*bert/encoder/layer_10/attention/self/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
(bert/encoder/layer_10/attention/self/subSub*bert/encoder/layer_10/attention/self/sub/x/bert/encoder/layer_10/attention/self/ExpandDims*(
_output_shapes
:??*
T0
q
,bert/encoder/layer_10/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
*bert/encoder/layer_10/attention/self/mul_1Mul(bert/encoder/layer_10/attention/self/sub,bert/encoder/layer_10/attention/self/mul_1/y*(
_output_shapes
:??*
T0
?
(bert/encoder/layer_10/attention/self/addAdd(bert/encoder/layer_10/attention/self/Mul*bert/encoder/layer_10/attention/self/mul_1*
T0*(
_output_shapes
:??
?
,bert/encoder/layer_10/attention/self/SoftmaxSoftmax(bert/encoder/layer_10/attention/self/add*
T0*(
_output_shapes
:??
?
4bert/encoder/layer_10/attention/self/Reshape_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?      @   
?
.bert/encoder/layer_10/attention/self/Reshape_2Reshape2bert/encoder/layer_10/attention/self/value/BiasAdd4bert/encoder/layer_10/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
5bert/encoder/layer_10/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
0bert/encoder/layer_10/attention/self/transpose_2	Transpose.bert/encoder/layer_10/attention/self/Reshape_25bert/encoder/layer_10/attention/self/transpose_2/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
-bert/encoder/layer_10/attention/self/MatMul_1BatchMatMul,bert/encoder/layer_10/attention/self/Softmax0bert/encoder/layer_10/attention/self/transpose_2*
T0*'
_output_shapes
:?@*
adj_x( *
adj_y( 
?
5bert/encoder/layer_10/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
0bert/encoder/layer_10/attention/self/transpose_3	Transpose-bert/encoder/layer_10/attention/self/MatMul_15bert/encoder/layer_10/attention/self/transpose_3/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
4bert/encoder/layer_10/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
.bert/encoder/layer_10/attention/self/Reshape_3Reshape0bert/encoder/layer_10/attention/self/transpose_34bert/encoder/layer_10/attention/self/Reshape_3/shape* 
_output_shapes
:
??*
T0*
Tshape0
?
Vbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
valueB"      *
dtype0
?
Ubert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Wbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
valueB
 *
ף<
?
`bert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalVbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/shape*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0
?
Tbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/mulMul`bert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalWbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel* 
_output_shapes
:
??
?
Pbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normalAddTbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/mulUbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel* 
_output_shapes
:
??
?
3bert/encoder/layer_10/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
	container *
shape:
??
?
:bert/encoder/layer_10/attention/output/dense/kernel/AssignAssign3bert/encoder/layer_10/attention/output/dense/kernelPbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel
?
8bert/encoder/layer_10/attention/output/dense/kernel/readIdentity3bert/encoder/layer_10/attention/output/dense/kernel* 
_output_shapes
:
??*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel
?
Cbert/encoder/layer_10/attention/output/dense/bias/Initializer/zerosConst*D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
1bert/encoder/layer_10/attention/output/dense/bias
VariableV2*
_output_shapes	
:?*
shared_name *D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias*
	container *
shape:?*
dtype0
?
8bert/encoder/layer_10/attention/output/dense/bias/AssignAssign1bert/encoder/layer_10/attention/output/dense/biasCbert/encoder/layer_10/attention/output/dense/bias/Initializer/zeros*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
6bert/encoder/layer_10/attention/output/dense/bias/readIdentity1bert/encoder/layer_10/attention/output/dense/bias*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias*
_output_shapes	
:?
?
3bert/encoder/layer_10/attention/output/dense/MatMulMatMul.bert/encoder/layer_10/attention/self/Reshape_38bert/encoder/layer_10/attention/output/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a( 
?
4bert/encoder/layer_10/attention/output/dense/BiasAddBiasAdd3bert/encoder/layer_10/attention/output/dense/MatMul6bert/encoder/layer_10/attention/output/dense/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
?
*bert/encoder/layer_10/attention/output/addAdd4bert/encoder/layer_10/attention/output/dense/BiasAdd5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
Gbert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zerosConst*
_output_shapes	
:?*H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
valueB?*    *
dtype0
?
5bert/encoder/layer_10/attention/output/LayerNorm/beta
VariableV2*
shared_name *H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
<bert/encoder/layer_10/attention/output/LayerNorm/beta/AssignAssign5bert/encoder/layer_10/attention/output/LayerNorm/betaGbert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
:bert/encoder/layer_10/attention/output/LayerNorm/beta/readIdentity5bert/encoder/layer_10/attention/output/LayerNorm/beta*H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
_output_shapes	
:?*
T0
?
Gbert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/onesConst*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
6bert/encoder/layer_10/attention/output/LayerNorm/gamma
VariableV2*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name 
?
=bert/encoder/layer_10/attention/output/LayerNorm/gamma/AssignAssign6bert/encoder/layer_10/attention/output/LayerNorm/gammaGbert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
;bert/encoder/layer_10/attention/output/LayerNorm/gamma/readIdentity6bert/encoder/layer_10/attention/output/LayerNorm/gamma*
T0*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
_output_shapes	
:?
?
Obert/encoder/layer_10/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
=bert/encoder/layer_10/attention/output/LayerNorm/moments/meanMean*bert/encoder/layer_10/attention/output/addObert/encoder/layer_10/attention/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
Ebert/encoder/layer_10/attention/output/LayerNorm/moments/StopGradientStopGradient=bert/encoder/layer_10/attention/output/LayerNorm/moments/mean*
_output_shapes
:	?*
T0
?
Jbert/encoder/layer_10/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference*bert/encoder/layer_10/attention/output/addEbert/encoder/layer_10/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Sbert/encoder/layer_10/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
?
Abert/encoder/layer_10/attention/output/LayerNorm/moments/varianceMeanJbert/encoder/layer_10/attention/output/LayerNorm/moments/SquaredDifferenceSbert/encoder/layer_10/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *̼?+
?
>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/addAddAbert/encoder/layer_10/attention/output/LayerNorm/moments/variance@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/RsqrtRsqrt>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mulMul@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/Rsqrt;bert/encoder/layer_10/attention/output/LayerNorm/gamma/read* 
_output_shapes
:
??*
T0
?
@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul_1Mul*bert/encoder/layer_10/attention/output/add>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul_2Mul=bert/encoder/layer_10/attention/output/LayerNorm/moments/mean>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/subSub:bert/encoder/layer_10/attention/output/LayerNorm/beta/read@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add_1Add@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul_1>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
??*
T0
?
Rbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Qbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Sbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
valueB
 *
ף<
?
\bert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/shape*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Pbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/mulMul\bert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalSbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel* 
_output_shapes
:
??
?
Lbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normalAddPbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/mulQbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel* 
_output_shapes
:
??
?
/bert/encoder/layer_10/intermediate/dense/kernel
VariableV2*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name 
?
6bert/encoder/layer_10/intermediate/dense/kernel/AssignAssign/bert/encoder/layer_10/intermediate/dense/kernelLbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
4bert/encoder/layer_10/intermediate/dense/kernel/readIdentity/bert/encoder/layer_10/intermediate/dense/kernel*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel* 
_output_shapes
:
??
?
Obert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
valueB:?*
dtype0*
_output_shapes
:
?
Ebert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros/ConstConst*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
?bert/encoder/layer_10/intermediate/dense/bias/Initializer/zerosFillObert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros/shape_as_tensorEbert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros/Const*
T0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*

index_type0*
_output_shapes	
:?
?
-bert/encoder/layer_10/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
	container *
shape:?
?
4bert/encoder/layer_10/intermediate/dense/bias/AssignAssign-bert/encoder/layer_10/intermediate/dense/bias?bert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
2bert/encoder/layer_10/intermediate/dense/bias/readIdentity-bert/encoder/layer_10/intermediate/dense/bias*
_output_shapes	
:?*
T0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias
?
/bert/encoder/layer_10/intermediate/dense/MatMulMatMul@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add_14bert/encoder/layer_10/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
0bert/encoder/layer_10/intermediate/dense/BiasAddBiasAdd/bert/encoder/layer_10/intermediate/dense/MatMul2bert/encoder/layer_10/intermediate/dense/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
s
.bert/encoder/layer_10/intermediate/dense/Pow/yConst*
_output_shapes
: *
valueB
 *  @@*
dtype0
?
,bert/encoder/layer_10/intermediate/dense/PowPow0bert/encoder/layer_10/intermediate/dense/BiasAdd.bert/encoder/layer_10/intermediate/dense/Pow/y*
T0* 
_output_shapes
:
??
s
.bert/encoder/layer_10/intermediate/dense/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *'7=
?
,bert/encoder/layer_10/intermediate/dense/mulMul.bert/encoder/layer_10/intermediate/dense/mul/x,bert/encoder/layer_10/intermediate/dense/Pow*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_10/intermediate/dense/addAdd0bert/encoder/layer_10/intermediate/dense/BiasAdd,bert/encoder/layer_10/intermediate/dense/mul*
T0* 
_output_shapes
:
??
u
0bert/encoder/layer_10/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
?
.bert/encoder/layer_10/intermediate/dense/mul_1Mul0bert/encoder/layer_10/intermediate/dense/mul_1/x,bert/encoder/layer_10/intermediate/dense/add* 
_output_shapes
:
??*
T0
?
-bert/encoder/layer_10/intermediate/dense/TanhTanh.bert/encoder/layer_10/intermediate/dense/mul_1* 
_output_shapes
:
??*
T0
u
0bert/encoder/layer_10/intermediate/dense/add_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
.bert/encoder/layer_10/intermediate/dense/add_1Add0bert/encoder/layer_10/intermediate/dense/add_1/x-bert/encoder/layer_10/intermediate/dense/Tanh*
T0* 
_output_shapes
:
??
u
0bert/encoder/layer_10/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
.bert/encoder/layer_10/intermediate/dense/mul_2Mul0bert/encoder/layer_10/intermediate/dense/mul_2/x.bert/encoder/layer_10/intermediate/dense/add_1*
T0* 
_output_shapes
:
??
?
.bert/encoder/layer_10/intermediate/dense/mul_3Mul0bert/encoder/layer_10/intermediate/dense/BiasAdd.bert/encoder/layer_10/intermediate/dense/mul_2* 
_output_shapes
:
??*
T0
?
Lbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/shapeConst*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Kbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/meanConst*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Mbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
valueB
 *
ף<*
dtype0
?
Vbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalLbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/shape*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Jbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/mulMulVbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalMbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel* 
_output_shapes
:
??
?
Fbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normalAddJbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/mulKbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/mean*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel* 
_output_shapes
:
??
?
)bert/encoder/layer_10/output/dense/kernel
VariableV2* 
_output_shapes
:
??*
shared_name *<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
	container *
shape:
??*
dtype0
?
0bert/encoder/layer_10/output/dense/kernel/AssignAssign)bert/encoder/layer_10/output/dense/kernelFbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel
?
.bert/encoder/layer_10/output/dense/kernel/readIdentity)bert/encoder/layer_10/output/dense/kernel*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel* 
_output_shapes
:
??
?
9bert/encoder/layer_10/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
valueB?*    
?
'bert/encoder/layer_10/output/dense/bias
VariableV2*
shared_name *:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
.bert/encoder/layer_10/output/dense/bias/AssignAssign'bert/encoder/layer_10/output/dense/bias9bert/encoder/layer_10/output/dense/bias/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
validate_shape(
?
,bert/encoder/layer_10/output/dense/bias/readIdentity'bert/encoder/layer_10/output/dense/bias*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
_output_shapes	
:?*
T0
?
)bert/encoder/layer_10/output/dense/MatMulMatMul.bert/encoder/layer_10/intermediate/dense/mul_3.bert/encoder/layer_10/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
*bert/encoder/layer_10/output/dense/BiasAddBiasAdd)bert/encoder/layer_10/output/dense/MatMul,bert/encoder/layer_10/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
 bert/encoder/layer_10/output/addAdd*bert/encoder/layer_10/output/dense/BiasAdd@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
valueB?*    
?
+bert/encoder/layer_10/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
	container *
shape:?
?
2bert/encoder/layer_10/output/LayerNorm/beta/AssignAssign+bert/encoder/layer_10/output/LayerNorm/beta=bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
0bert/encoder/layer_10/output/LayerNorm/beta/readIdentity+bert/encoder/layer_10/output/LayerNorm/beta*
T0*>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
_output_shapes	
:?
?
=bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/onesConst*
_output_shapes	
:?*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
valueB?*  ??*
dtype0
?
,bert/encoder/layer_10/output/LayerNorm/gamma
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
	container 
?
3bert/encoder/layer_10/output/LayerNorm/gamma/AssignAssign,bert/encoder/layer_10/output/LayerNorm/gamma=bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma
?
1bert/encoder/layer_10/output/LayerNorm/gamma/readIdentity,bert/encoder/layer_10/output/LayerNorm/gamma*
T0*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
_output_shapes	
:?
?
Ebert/encoder/layer_10/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
3bert/encoder/layer_10/output/LayerNorm/moments/meanMean bert/encoder/layer_10/output/addEbert/encoder/layer_10/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	?*
	keep_dims(*

Tidx0*
T0
?
;bert/encoder/layer_10/output/LayerNorm/moments/StopGradientStopGradient3bert/encoder/layer_10/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
@bert/encoder/layer_10/output/LayerNorm/moments/SquaredDifferenceSquaredDifference bert/encoder/layer_10/output/add;bert/encoder/layer_10/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Ibert/encoder/layer_10/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
7bert/encoder/layer_10/output/LayerNorm/moments/varianceMean@bert/encoder/layer_10/output/LayerNorm/moments/SquaredDifferenceIbert/encoder/layer_10/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
{
6bert/encoder/layer_10/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
4bert/encoder/layer_10/output/LayerNorm/batchnorm/addAdd7bert/encoder/layer_10/output/LayerNorm/moments/variance6bert/encoder/layer_10/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
6bert/encoder/layer_10/output/LayerNorm/batchnorm/RsqrtRsqrt4bert/encoder/layer_10/output/LayerNorm/batchnorm/add*
_output_shapes
:	?*
T0
?
4bert/encoder/layer_10/output/LayerNorm/batchnorm/mulMul6bert/encoder/layer_10/output/LayerNorm/batchnorm/Rsqrt1bert/encoder/layer_10/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
6bert/encoder/layer_10/output/LayerNorm/batchnorm/mul_1Mul bert/encoder/layer_10/output/add4bert/encoder/layer_10/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
6bert/encoder/layer_10/output/LayerNorm/batchnorm/mul_2Mul3bert/encoder/layer_10/output/LayerNorm/moments/mean4bert/encoder/layer_10/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
4bert/encoder/layer_10/output/LayerNorm/batchnorm/subSub0bert/encoder/layer_10/output/LayerNorm/beta/read6bert/encoder/layer_10/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_1Add6bert/encoder/layer_10/output/LayerNorm/batchnorm/mul_14bert/encoder/layer_10/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Tbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Sbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/meanConst*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ubert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
^bert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/shape*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Rbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/mulMul^bert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalUbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel* 
_output_shapes
:
??
?
Nbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normalAddRbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/mulSbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel* 
_output_shapes
:
??
?
1bert/encoder/layer_11/attention/self/query/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
	container *
shape:
??
?
8bert/encoder/layer_11/attention/self/query/kernel/AssignAssign1bert/encoder/layer_11/attention/self/query/kernelNbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
6bert/encoder/layer_11/attention/self/query/kernel/readIdentity1bert/encoder/layer_11/attention/self/query/kernel* 
_output_shapes
:
??*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel
?
Abert/encoder/layer_11/attention/self/query/bias/Initializer/zerosConst*B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
/bert/encoder/layer_11/attention/self/query/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
	container 
?
6bert/encoder/layer_11/attention/self/query/bias/AssignAssign/bert/encoder/layer_11/attention/self/query/biasAbert/encoder/layer_11/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
4bert/encoder/layer_11/attention/self/query/bias/readIdentity/bert/encoder/layer_11/attention/self/query/bias*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
_output_shapes	
:?
?
1bert/encoder/layer_11/attention/self/query/MatMulMatMul6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_16bert/encoder/layer_11/attention/self/query/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
2bert/encoder/layer_11/attention/self/query/BiasAddBiasAdd1bert/encoder/layer_11/attention/self/query/MatMul4bert/encoder/layer_11/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
Rbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Qbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/meanConst*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Sbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
\bert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/shape*

seed *
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
??
?
Pbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/mulMul\bert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalSbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel* 
_output_shapes
:
??
?
Lbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normalAddPbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/mulQbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel* 
_output_shapes
:
??
?
/bert/encoder/layer_11/attention/self/key/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
	container *
shape:
??
?
6bert/encoder/layer_11/attention/self/key/kernel/AssignAssign/bert/encoder/layer_11/attention/self/key/kernelLbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
4bert/encoder/layer_11/attention/self/key/kernel/readIdentity/bert/encoder/layer_11/attention/self/key/kernel* 
_output_shapes
:
??*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel
?
?bert/encoder/layer_11/attention/self/key/bias/Initializer/zerosConst*@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
-bert/encoder/layer_11/attention/self/key/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias
?
4bert/encoder/layer_11/attention/self/key/bias/AssignAssign-bert/encoder/layer_11/attention/self/key/bias?bert/encoder/layer_11/attention/self/key/bias/Initializer/zeros*
T0*@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
2bert/encoder/layer_11/attention/self/key/bias/readIdentity-bert/encoder/layer_11/attention/self/key/bias*
T0*@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias*
_output_shapes	
:?
?
/bert/encoder/layer_11/attention/self/key/MatMulMatMul6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_14bert/encoder/layer_11/attention/self/key/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
0bert/encoder/layer_11/attention/self/key/BiasAddBiasAdd/bert/encoder/layer_11/attention/self/key/MatMul2bert/encoder/layer_11/attention/self/key/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
Tbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Sbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/meanConst*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ubert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
valueB
 *
ף<
?
^bert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel
?
Rbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/mulMul^bert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalUbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/stddev*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel* 
_output_shapes
:
??*
T0
?
Nbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normalAddRbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/mulSbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel
?
1bert/encoder/layer_11/attention/self/value/kernel
VariableV2*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel
?
8bert/encoder/layer_11/attention/self/value/kernel/AssignAssign1bert/encoder/layer_11/attention/self/value/kernelNbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
6bert/encoder/layer_11/attention/self/value/kernel/readIdentity1bert/encoder/layer_11/attention/self/value/kernel*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel* 
_output_shapes
:
??
?
Abert/encoder/layer_11/attention/self/value/bias/Initializer/zerosConst*B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
/bert/encoder/layer_11/attention/self/value/bias
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
	container 
?
6bert/encoder/layer_11/attention/self/value/bias/AssignAssign/bert/encoder/layer_11/attention/self/value/biasAbert/encoder/layer_11/attention/self/value/bias/Initializer/zeros*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
4bert/encoder/layer_11/attention/self/value/bias/readIdentity/bert/encoder/layer_11/attention/self/value/bias*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
_output_shapes	
:?
?
1bert/encoder/layer_11/attention/self/value/MatMulMatMul6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_16bert/encoder/layer_11/attention/self/value/kernel/read* 
_output_shapes
:
??*
transpose_a( *
transpose_b( *
T0
?
2bert/encoder/layer_11/attention/self/value/BiasAddBiasAdd1bert/encoder/layer_11/attention/self/value/MatMul4bert/encoder/layer_11/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
??*
T0
?
2bert/encoder/layer_11/attention/self/Reshape/shapeConst*%
valueB"   ?      @   *
dtype0*
_output_shapes
:
?
,bert/encoder/layer_11/attention/self/ReshapeReshape2bert/encoder/layer_11/attention/self/query/BiasAdd2bert/encoder/layer_11/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
3bert/encoder/layer_11/attention/self/transpose/permConst*
_output_shapes
:*%
valueB"             *
dtype0
?
.bert/encoder/layer_11/attention/self/transpose	Transpose,bert/encoder/layer_11/attention/self/Reshape3bert/encoder/layer_11/attention/self/transpose/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
4bert/encoder/layer_11/attention/self/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?      @   
?
.bert/encoder/layer_11/attention/self/Reshape_1Reshape0bert/encoder/layer_11/attention/self/key/BiasAdd4bert/encoder/layer_11/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
5bert/encoder/layer_11/attention/self/transpose_1/permConst*
_output_shapes
:*%
valueB"             *
dtype0
?
0bert/encoder/layer_11/attention/self/transpose_1	Transpose.bert/encoder/layer_11/attention/self/Reshape_15bert/encoder/layer_11/attention/self/transpose_1/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
+bert/encoder/layer_11/attention/self/MatMulBatchMatMul.bert/encoder/layer_11/attention/self/transpose0bert/encoder/layer_11/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:??
o
*bert/encoder/layer_11/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
?
(bert/encoder/layer_11/attention/self/MulMul+bert/encoder/layer_11/attention/self/MatMul*bert/encoder/layer_11/attention/self/Mul/y*
T0*(
_output_shapes
:??
}
3bert/encoder/layer_11/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
?
/bert/encoder/layer_11/attention/self/ExpandDims
ExpandDimsbert/encoder/mul3bert/encoder/layer_11/attention/self/ExpandDims/dim*(
_output_shapes
:??*

Tdim0*
T0
o
*bert/encoder/layer_11/attention/self/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
(bert/encoder/layer_11/attention/self/subSub*bert/encoder/layer_11/attention/self/sub/x/bert/encoder/layer_11/attention/self/ExpandDims*(
_output_shapes
:??*
T0
q
,bert/encoder/layer_11/attention/self/mul_1/yConst*
valueB
 * @?*
dtype0*
_output_shapes
: 
?
*bert/encoder/layer_11/attention/self/mul_1Mul(bert/encoder/layer_11/attention/self/sub,bert/encoder/layer_11/attention/self/mul_1/y*(
_output_shapes
:??*
T0
?
(bert/encoder/layer_11/attention/self/addAdd(bert/encoder/layer_11/attention/self/Mul*bert/encoder/layer_11/attention/self/mul_1*(
_output_shapes
:??*
T0
?
,bert/encoder/layer_11/attention/self/SoftmaxSoftmax(bert/encoder/layer_11/attention/self/add*(
_output_shapes
:??*
T0
?
4bert/encoder/layer_11/attention/self/Reshape_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"   ?      @   
?
.bert/encoder/layer_11/attention/self/Reshape_2Reshape2bert/encoder/layer_11/attention/self/value/BiasAdd4bert/encoder/layer_11/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:?@
?
5bert/encoder/layer_11/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
0bert/encoder/layer_11/attention/self/transpose_2	Transpose.bert/encoder/layer_11/attention/self/Reshape_25bert/encoder/layer_11/attention/self/transpose_2/perm*
T0*'
_output_shapes
:?@*
Tperm0
?
-bert/encoder/layer_11/attention/self/MatMul_1BatchMatMul,bert/encoder/layer_11/attention/self/Softmax0bert/encoder/layer_11/attention/self/transpose_2*'
_output_shapes
:?@*
adj_x( *
adj_y( *
T0
?
5bert/encoder/layer_11/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
?
0bert/encoder/layer_11/attention/self/transpose_3	Transpose-bert/encoder/layer_11/attention/self/MatMul_15bert/encoder/layer_11/attention/self/transpose_3/perm*'
_output_shapes
:?@*
Tperm0*
T0
?
4bert/encoder/layer_11/attention/self/Reshape_3/shapeConst*
valueB"?      *
dtype0*
_output_shapes
:
?
.bert/encoder/layer_11/attention/self/Reshape_3Reshape0bert/encoder/layer_11/attention/self/transpose_34bert/encoder/layer_11/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:
??
?
Vbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Ubert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Wbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
`bert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalVbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/shape*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??*

seed 
?
Tbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/mulMul`bert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalWbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
??*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel
?
Pbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normalAddTbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/mulUbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
??*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel
?
3bert/encoder/layer_11/attention/output/dense/kernel
VariableV2*
	container *
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel
?
:bert/encoder/layer_11/attention/output/dense/kernel/AssignAssign3bert/encoder/layer_11/attention/output/dense/kernelPbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
8bert/encoder/layer_11/attention/output/dense/kernel/readIdentity3bert/encoder/layer_11/attention/output/dense/kernel* 
_output_shapes
:
??*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel
?
Cbert/encoder/layer_11/attention/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias*
valueB?*    
?
1bert/encoder/layer_11/attention/output/dense/bias
VariableV2*
shared_name *D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
8bert/encoder/layer_11/attention/output/dense/bias/AssignAssign1bert/encoder/layer_11/attention/output/dense/biasCbert/encoder/layer_11/attention/output/dense/bias/Initializer/zeros*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
6bert/encoder/layer_11/attention/output/dense/bias/readIdentity1bert/encoder/layer_11/attention/output/dense/bias*
_output_shapes	
:?*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias
?
3bert/encoder/layer_11/attention/output/dense/MatMulMatMul.bert/encoder/layer_11/attention/self/Reshape_38bert/encoder/layer_11/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
4bert/encoder/layer_11/attention/output/dense/BiasAddBiasAdd3bert/encoder/layer_11/attention/output/dense/MatMul6bert/encoder/layer_11/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
?
*bert/encoder/layer_11/attention/output/addAdd4bert/encoder/layer_11/attention/output/dense/BiasAdd6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
Gbert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zerosConst*H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
5bert/encoder/layer_11/attention/output/LayerNorm/beta
VariableV2*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
	container 
?
<bert/encoder/layer_11/attention/output/LayerNorm/beta/AssignAssign5bert/encoder/layer_11/attention/output/LayerNorm/betaGbert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
:bert/encoder/layer_11/attention/output/LayerNorm/beta/readIdentity5bert/encoder/layer_11/attention/output/LayerNorm/beta*H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
_output_shapes	
:?*
T0
?
Gbert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/onesConst*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
6bert/encoder/layer_11/attention/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
	container *
shape:?
?
=bert/encoder/layer_11/attention/output/LayerNorm/gamma/AssignAssign6bert/encoder/layer_11/attention/output/LayerNorm/gammaGbert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/ones*
T0*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
;bert/encoder/layer_11/attention/output/LayerNorm/gamma/readIdentity6bert/encoder/layer_11/attention/output/LayerNorm/gamma*
T0*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
_output_shapes	
:?
?
Obert/encoder/layer_11/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
=bert/encoder/layer_11/attention/output/LayerNorm/moments/meanMean*bert/encoder/layer_11/attention/output/addObert/encoder/layer_11/attention/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
Ebert/encoder/layer_11/attention/output/LayerNorm/moments/StopGradientStopGradient=bert/encoder/layer_11/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
Jbert/encoder/layer_11/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference*bert/encoder/layer_11/attention/output/addEbert/encoder/layer_11/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Sbert/encoder/layer_11/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
Abert/encoder/layer_11/attention/output/LayerNorm/moments/varianceMeanJbert/encoder/layer_11/attention/output/LayerNorm/moments/SquaredDifferenceSbert/encoder/layer_11/attention/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *̼?+*
dtype0*
_output_shapes
: 
?
>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/addAddAbert/encoder/layer_11/attention/output/LayerNorm/moments/variance@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/RsqrtRsqrt>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mulMul@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/Rsqrt;bert/encoder/layer_11/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul_1Mul*bert/encoder/layer_11/attention/output/add>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul_2Mul=bert/encoder/layer_11/attention/output/LayerNorm/moments/mean>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:
??
?
>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/subSub:bert/encoder/layer_11/attention/output/LayerNorm/beta/read@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add_1Add@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul_1>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
?
Rbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Qbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Sbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
valueB
 *
ף<
?
\bert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
??*

seed *
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
seed2 
?
Pbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/mulMul\bert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalSbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/stddev*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel* 
_output_shapes
:
??*
T0
?
Lbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normalAddPbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/mulQbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel* 
_output_shapes
:
??
?
/bert/encoder/layer_11/intermediate/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
	container *
shape:
??
?
6bert/encoder/layer_11/intermediate/dense/kernel/AssignAssign/bert/encoder/layer_11/intermediate/dense/kernelLbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
4bert/encoder/layer_11/intermediate/dense/kernel/readIdentity/bert/encoder/layer_11/intermediate/dense/kernel*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel* 
_output_shapes
:
??*
T0
?
Obert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
valueB:?*
dtype0*
_output_shapes
:
?
Ebert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros/ConstConst*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
?
?bert/encoder/layer_11/intermediate/dense/bias/Initializer/zerosFillObert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros/shape_as_tensorEbert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros/Const*
T0*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*

index_type0*
_output_shapes	
:?
?
-bert/encoder/layer_11/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
	container *
shape:?
?
4bert/encoder/layer_11/intermediate/dense/bias/AssignAssign-bert/encoder/layer_11/intermediate/dense/bias?bert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
2bert/encoder/layer_11/intermediate/dense/bias/readIdentity-bert/encoder/layer_11/intermediate/dense/bias*
T0*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
_output_shapes	
:?
?
/bert/encoder/layer_11/intermediate/dense/MatMulMatMul@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add_14bert/encoder/layer_11/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
0bert/encoder/layer_11/intermediate/dense/BiasAddBiasAdd/bert/encoder/layer_11/intermediate/dense/MatMul2bert/encoder/layer_11/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
??
s
.bert/encoder/layer_11/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
?
,bert/encoder/layer_11/intermediate/dense/PowPow0bert/encoder/layer_11/intermediate/dense/BiasAdd.bert/encoder/layer_11/intermediate/dense/Pow/y* 
_output_shapes
:
??*
T0
s
.bert/encoder/layer_11/intermediate/dense/mul/xConst*
_output_shapes
: *
valueB
 *'7=*
dtype0
?
,bert/encoder/layer_11/intermediate/dense/mulMul.bert/encoder/layer_11/intermediate/dense/mul/x,bert/encoder/layer_11/intermediate/dense/Pow*
T0* 
_output_shapes
:
??
?
,bert/encoder/layer_11/intermediate/dense/addAdd0bert/encoder/layer_11/intermediate/dense/BiasAdd,bert/encoder/layer_11/intermediate/dense/mul*
T0* 
_output_shapes
:
??
u
0bert/encoder/layer_11/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
?
.bert/encoder/layer_11/intermediate/dense/mul_1Mul0bert/encoder/layer_11/intermediate/dense/mul_1/x,bert/encoder/layer_11/intermediate/dense/add* 
_output_shapes
:
??*
T0
?
-bert/encoder/layer_11/intermediate/dense/TanhTanh.bert/encoder/layer_11/intermediate/dense/mul_1*
T0* 
_output_shapes
:
??
u
0bert/encoder/layer_11/intermediate/dense/add_1/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
.bert/encoder/layer_11/intermediate/dense/add_1Add0bert/encoder/layer_11/intermediate/dense/add_1/x-bert/encoder/layer_11/intermediate/dense/Tanh*
T0* 
_output_shapes
:
??
u
0bert/encoder/layer_11/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
.bert/encoder/layer_11/intermediate/dense/mul_2Mul0bert/encoder/layer_11/intermediate/dense/mul_2/x.bert/encoder/layer_11/intermediate/dense/add_1* 
_output_shapes
:
??*
T0
?
.bert/encoder/layer_11/intermediate/dense/mul_3Mul0bert/encoder/layer_11/intermediate/dense/BiasAdd.bert/encoder/layer_11/intermediate/dense/mul_2*
T0* 
_output_shapes
:
??
?
Lbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/shapeConst*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
Kbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/meanConst*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Mbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/stddevConst*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
valueB
 *
ף<*
dtype0*
_output_shapes
: 
?
Vbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalLbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/shape*

seed *
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
??
?
Jbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/mulMulVbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalMbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel* 
_output_shapes
:
??
?
Fbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normalAddJbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/mulKbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/mean*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel* 
_output_shapes
:
??
?
)bert/encoder/layer_11/output/dense/kernel
VariableV2*
shape:
??*
dtype0* 
_output_shapes
:
??*
shared_name *<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
	container 
?
0bert/encoder/layer_11/output/dense/kernel/AssignAssign)bert/encoder/layer_11/output/dense/kernelFbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
.bert/encoder/layer_11/output/dense/kernel/readIdentity)bert/encoder/layer_11/output/dense/kernel* 
_output_shapes
:
??*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel
?
9bert/encoder/layer_11/output/dense/bias/Initializer/zerosConst*:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
'bert/encoder/layer_11/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
	container *
shape:?
?
.bert/encoder/layer_11/output/dense/bias/AssignAssign'bert/encoder/layer_11/output/dense/bias9bert/encoder/layer_11/output/dense/bias/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
,bert/encoder/layer_11/output/dense/bias/readIdentity'bert/encoder/layer_11/output/dense/bias*
T0*:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
_output_shapes	
:?
?
)bert/encoder/layer_11/output/dense/MatMulMatMul.bert/encoder/layer_11/intermediate/dense/mul_3.bert/encoder/layer_11/output/dense/kernel/read*
T0* 
_output_shapes
:
??*
transpose_a( *
transpose_b( 
?
*bert/encoder/layer_11/output/dense/BiasAddBiasAdd)bert/encoder/layer_11/output/dense/MatMul,bert/encoder/layer_11/output/dense/bias/read* 
_output_shapes
:
??*
T0*
data_formatNHWC
?
 bert/encoder/layer_11/output/addAdd*bert/encoder/layer_11/output/dense/BiasAdd@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:
??
?
=bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zerosConst*
_output_shapes	
:?*>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
valueB?*    *
dtype0
?
+bert/encoder/layer_11/output/LayerNorm/beta
VariableV2*
_output_shapes	
:?*
shared_name *>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
	container *
shape:?*
dtype0
?
2bert/encoder/layer_11/output/LayerNorm/beta/AssignAssign+bert/encoder/layer_11/output/LayerNorm/beta=bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zeros*>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
0bert/encoder/layer_11/output/LayerNorm/beta/readIdentity+bert/encoder/layer_11/output/LayerNorm/beta*
_output_shapes	
:?*
T0*>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta
?
=bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/onesConst*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
,bert/encoder/layer_11/output/LayerNorm/gamma
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
3bert/encoder/layer_11/output/LayerNorm/gamma/AssignAssign,bert/encoder/layer_11/output/LayerNorm/gamma=bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/ones*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
1bert/encoder/layer_11/output/LayerNorm/gamma/readIdentity,bert/encoder/layer_11/output/LayerNorm/gamma*
T0*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
_output_shapes	
:?
?
Ebert/encoder/layer_11/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
3bert/encoder/layer_11/output/LayerNorm/moments/meanMean bert/encoder/layer_11/output/addEbert/encoder/layer_11/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	?
?
;bert/encoder/layer_11/output/LayerNorm/moments/StopGradientStopGradient3bert/encoder/layer_11/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	?
?
@bert/encoder/layer_11/output/LayerNorm/moments/SquaredDifferenceSquaredDifference bert/encoder/layer_11/output/add;bert/encoder/layer_11/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:
??
?
Ibert/encoder/layer_11/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
?
7bert/encoder/layer_11/output/LayerNorm/moments/varianceMean@bert/encoder/layer_11/output/LayerNorm/moments/SquaredDifferenceIbert/encoder/layer_11/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	?*
	keep_dims(*

Tidx0
{
6bert/encoder/layer_11/output/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *̼?+*
dtype0
?
4bert/encoder/layer_11/output/LayerNorm/batchnorm/addAdd7bert/encoder/layer_11/output/LayerNorm/moments/variance6bert/encoder/layer_11/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	?
?
6bert/encoder/layer_11/output/LayerNorm/batchnorm/RsqrtRsqrt4bert/encoder/layer_11/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	?
?
4bert/encoder/layer_11/output/LayerNorm/batchnorm/mulMul6bert/encoder/layer_11/output/LayerNorm/batchnorm/Rsqrt1bert/encoder/layer_11/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:
??
?
6bert/encoder/layer_11/output/LayerNorm/batchnorm/mul_1Mul bert/encoder/layer_11/output/add4bert/encoder/layer_11/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
6bert/encoder/layer_11/output/LayerNorm/batchnorm/mul_2Mul3bert/encoder/layer_11/output/LayerNorm/moments/mean4bert/encoder/layer_11/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
??*
T0
?
4bert/encoder/layer_11/output/LayerNorm/batchnorm/subSub0bert/encoder/layer_11/output/LayerNorm/beta/read6bert/encoder/layer_11/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:
??
?
6bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1Add6bert/encoder/layer_11/output/LayerNorm/batchnorm/mul_14bert/encoder/layer_11/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:
??
q
bert/encoder/Reshape_2/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_2Reshape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_2/shape*$
_output_shapes
:??*
T0*
Tshape0
q
bert/encoder/Reshape_3/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_3Reshape5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_3/shape*
T0*
Tshape0*$
_output_shapes
:??
q
bert/encoder/Reshape_4/shapeConst*
dtype0*
_output_shapes
:*!
valueB"   ?      
?
bert/encoder/Reshape_4Reshape5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_4/shape*$
_output_shapes
:??*
T0*
Tshape0
q
bert/encoder/Reshape_5/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_5Reshape5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_5/shape*
Tshape0*$
_output_shapes
:??*
T0
q
bert/encoder/Reshape_6/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_6Reshape5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_6/shape*
T0*
Tshape0*$
_output_shapes
:??
q
bert/encoder/Reshape_7/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_7Reshape5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_7/shape*
Tshape0*$
_output_shapes
:??*
T0
q
bert/encoder/Reshape_8/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_8Reshape5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_8/shape*$
_output_shapes
:??*
T0*
Tshape0
q
bert/encoder/Reshape_9/shapeConst*
_output_shapes
:*!
valueB"   ?      *
dtype0
?
bert/encoder/Reshape_9Reshape5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_9/shape*$
_output_shapes
:??*
T0*
Tshape0
r
bert/encoder/Reshape_10/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_10Reshape5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_10/shape*$
_output_shapes
:??*
T0*
Tshape0
r
bert/encoder/Reshape_11/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_11Reshape5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_11/shape*$
_output_shapes
:??*
T0*
Tshape0
r
bert/encoder/Reshape_12/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_12Reshape6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_12/shape*
T0*
Tshape0*$
_output_shapes
:??
r
bert/encoder/Reshape_13/shapeConst*!
valueB"   ?      *
dtype0*
_output_shapes
:
?
bert/encoder/Reshape_13Reshape6bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_13/shape*$
_output_shapes
:??*
T0*
Tshape0
t
bert/pooler/strided_slice/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
v
!bert/pooler/strided_slice/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
v
!bert/pooler/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
?
bert/pooler/strided_sliceStridedSlicebert/encoder/Reshape_13bert/pooler/strided_slice/stack!bert/pooler/strided_slice/stack_1!bert/pooler/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:?*
T0*
Index0*
shrink_axis_mask 
z
bert/pooler/SqueezeSqueezebert/pooler/strided_slice*
squeeze_dims
*
T0*
_output_shapes
:	?
?
;bert/pooler/dense/kernel/Initializer/truncated_normal/shapeConst*+
_class!
loc:@bert/pooler/dense/kernel*
valueB"      *
dtype0*
_output_shapes
:
?
:bert/pooler/dense/kernel/Initializer/truncated_normal/meanConst*+
_class!
loc:@bert/pooler/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
?
<bert/pooler/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *+
_class!
loc:@bert/pooler/dense/kernel*
valueB
 *
ף<
?
Ebert/pooler/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;bert/pooler/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
??*

seed *
T0*+
_class!
loc:@bert/pooler/dense/kernel
?
9bert/pooler/dense/kernel/Initializer/truncated_normal/mulMulEbert/pooler/dense/kernel/Initializer/truncated_normal/TruncatedNormal<bert/pooler/dense/kernel/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:
??
?
5bert/pooler/dense/kernel/Initializer/truncated_normalAdd9bert/pooler/dense/kernel/Initializer/truncated_normal/mul:bert/pooler/dense/kernel/Initializer/truncated_normal/mean*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:
??*
T0
?
bert/pooler/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
??*
shared_name *+
_class!
loc:@bert/pooler/dense/kernel*
	container *
shape:
??
?
bert/pooler/dense/kernel/AssignAssignbert/pooler/dense/kernel5bert/pooler/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*+
_class!
loc:@bert/pooler/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
bert/pooler/dense/kernel/readIdentitybert/pooler/dense/kernel*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:
??
?
(bert/pooler/dense/bias/Initializer/zerosConst*)
_class
loc:@bert/pooler/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
bert/pooler/dense/bias
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *)
_class
loc:@bert/pooler/dense/bias
?
bert/pooler/dense/bias/AssignAssignbert/pooler/dense/bias(bert/pooler/dense/bias/Initializer/zeros*)
_class
loc:@bert/pooler/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
bert/pooler/dense/bias/readIdentitybert/pooler/dense/bias*
_output_shapes	
:?*
T0*)
_class
loc:@bert/pooler/dense/bias
?
bert/pooler/dense/MatMulMatMulbert/pooler/Squeezebert/pooler/dense/kernel/read*
T0*
_output_shapes
:	?*
transpose_a( *
transpose_b( 
?
bert/pooler/dense/BiasAddBiasAddbert/pooler/dense/MatMulbert/pooler/dense/bias/read*
data_formatNHWC*
_output_shapes
:	?*
T0
c
bert/pooler/dense/TanhTanhbert/pooler/dense/BiasAdd*
_output_shapes
:	?*
T0
?
1output_weights/Initializer/truncated_normal/shapeConst*!
_class
loc:@output_weights*
valueB"      *
dtype0*
_output_shapes
:
?
0output_weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *!
_class
loc:@output_weights*
valueB
 *    
?
2output_weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *!
_class
loc:@output_weights*
valueB
 *
ף<
?
;output_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal1output_weights/Initializer/truncated_normal/shape*
dtype0*
_output_shapes
:	?*

seed *
T0*!
_class
loc:@output_weights*
seed2 
?
/output_weights/Initializer/truncated_normal/mulMul;output_weights/Initializer/truncated_normal/TruncatedNormal2output_weights/Initializer/truncated_normal/stddev*
_output_shapes
:	?*
T0*!
_class
loc:@output_weights
?
+output_weights/Initializer/truncated_normalAdd/output_weights/Initializer/truncated_normal/mul0output_weights/Initializer/truncated_normal/mean*
_output_shapes
:	?*
T0*!
_class
loc:@output_weights
?
output_weights
VariableV2*
dtype0*
_output_shapes
:	?*
shared_name *!
_class
loc:@output_weights*
	container *
shape:	?
?
output_weights/AssignAssignoutput_weights+output_weights/Initializer/truncated_normal*
validate_shape(*
_output_shapes
:	?*
use_locking(*
T0*!
_class
loc:@output_weights
|
output_weights/readIdentityoutput_weights*
T0*!
_class
loc:@output_weights*
_output_shapes
:	?
?
output_bias/Initializer/zerosConst*
_class
loc:@output_bias*
valueB*    *
dtype0*
_output_shapes
:
?
output_bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@output_bias*
	container *
shape:
?
output_bias/AssignAssignoutput_biasoutput_bias/Initializer/zeros*
T0*
_class
loc:@output_bias*
validate_shape(*
_output_shapes
:*
use_locking(
n
output_bias/readIdentityoutput_bias*
T0*
_class
loc:@output_bias*
_output_shapes
:
?
loss/MatMulMatMulbert/pooler/dense/Tanhoutput_weights/read*
_output_shapes

:*
transpose_a( *
transpose_b(*
T0
v
loss/BiasAddBiasAddloss/MatMuloutput_bias/read*
T0*
data_formatNHWC*
_output_shapes

:
N
loss/SigmoidSigmoidloss/BiasAdd*
_output_shapes

:*
T0
a
loss/SqueezeSqueezeloss/Sigmoid*
T0*
_output_shapes
:*
squeeze_dims

w
)loss/mean_squared_error/SquaredDifferenceSquaredDifferenceloss/Squeezevals*
_output_shapes
:*
T0
y
4loss/mean_squared_error/assert_broadcastable/weightsConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
}
:loss/mean_squared_error/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
{
9loss/mean_squared_error/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
?
9loss/mean_squared_error/assert_broadcastable/values/shapeConst*
dtype0*
_output_shapes
:*
valueB:
z
8loss/mean_squared_error/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
P
Hloss/mean_squared_error/assert_broadcastable/static_scalar_check_successNoOp
?
!loss/mean_squared_error/ToFloat/xConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
loss/mean_squared_error/MulMul)loss/mean_squared_error/SquaredDifference!loss/mean_squared_error/ToFloat/x*
_output_shapes
:*
T0
?
loss/mean_squared_error/ConstConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
?
loss/mean_squared_error/SumSumloss/mean_squared_error/Mulloss/mean_squared_error/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
?
+loss/mean_squared_error/num_present/Equal/yConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
?
)loss/mean_squared_error/num_present/EqualEqual!loss/mean_squared_error/ToFloat/x+loss/mean_squared_error/num_present/Equal/y*
_output_shapes
: *
T0
?
.loss/mean_squared_error/num_present/zeros_likeConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
?
3loss/mean_squared_error/num_present/ones_like/ShapeConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
3loss/mean_squared_error/num_present/ones_like/ConstConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
-loss/mean_squared_error/num_present/ones_likeFill3loss/mean_squared_error/num_present/ones_like/Shape3loss/mean_squared_error/num_present/ones_like/Const*
_output_shapes
: *
T0*

index_type0
?
*loss/mean_squared_error/num_present/SelectSelect)loss/mean_squared_error/num_present/Equal.loss/mean_squared_error/num_present/zeros_like-loss/mean_squared_error/num_present/ones_like*
_output_shapes
: *
T0
?
Xloss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
Wloss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B : *
dtype0*
_output_shapes
: 
?
Wloss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
?
Vloss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
?
floss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success
?
Eloss/mean_squared_error/num_present/broadcast_weights/ones_like/ShapeConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_successg^loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB:*
dtype0*
_output_shapes
:
?
Eloss/mean_squared_error/num_present/broadcast_weights/ones_like/ConstConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_successg^loss/mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
?loss/mean_squared_error/num_present/broadcast_weights/ones_likeFillEloss/mean_squared_error/num_present/broadcast_weights/ones_like/ShapeEloss/mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*
_output_shapes
:
?
5loss/mean_squared_error/num_present/broadcast_weightsMul*loss/mean_squared_error/num_present/Select?loss/mean_squared_error/num_present/broadcast_weights/ones_like*
_output_shapes
:*
T0
?
)loss/mean_squared_error/num_present/ConstConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB: *
dtype0*
_output_shapes
:
?
#loss/mean_squared_error/num_presentSum5loss/mean_squared_error/num_present/broadcast_weights)loss/mean_squared_error/num_present/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
?
loss/mean_squared_error/Const_1ConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
loss/mean_squared_error/Sum_1Sumloss/mean_squared_error/Sumloss/mean_squared_error/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
?
!loss/mean_squared_error/Greater/yConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
?
loss/mean_squared_error/GreaterGreater#loss/mean_squared_error/num_present!loss/mean_squared_error/Greater/y*
_output_shapes
: *
T0
?
loss/mean_squared_error/Equal/yConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
?
loss/mean_squared_error/EqualEqual#loss/mean_squared_error/num_presentloss/mean_squared_error/Equal/y*
_output_shapes
: *
T0
?
'loss/mean_squared_error/ones_like/ShapeConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
?
'loss/mean_squared_error/ones_like/ConstConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
!loss/mean_squared_error/ones_likeFill'loss/mean_squared_error/ones_like/Shape'loss/mean_squared_error/ones_like/Const*
_output_shapes
: *
T0*

index_type0
?
loss/mean_squared_error/SelectSelectloss/mean_squared_error/Equal!loss/mean_squared_error/ones_like#loss/mean_squared_error/num_present*
_output_shapes
: *
T0
?
loss/mean_squared_error/divRealDivloss/mean_squared_error/Sum_1loss/mean_squared_error/Select*
_output_shapes
: *
T0
?
"loss/mean_squared_error/zeros_likeConstI^loss/mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
?
loss/mean_squared_error/valueSelectloss/mean_squared_error/Greaterloss/mean_squared_error/div"loss/mean_squared_error/zeros_like*
_output_shapes
: *
T0
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
?J
save/SaveV2/tensor_namesConst*?J
value?JB?J?Bbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelB5bert/encoder/layer_10/attention/output/LayerNorm/betaB6bert/encoder/layer_10/attention/output/LayerNorm/gammaB1bert/encoder/layer_10/attention/output/dense/biasB3bert/encoder/layer_10/attention/output/dense/kernelB-bert/encoder/layer_10/attention/self/key/biasB/bert/encoder/layer_10/attention/self/key/kernelB/bert/encoder/layer_10/attention/self/query/biasB1bert/encoder/layer_10/attention/self/query/kernelB/bert/encoder/layer_10/attention/self/value/biasB1bert/encoder/layer_10/attention/self/value/kernelB-bert/encoder/layer_10/intermediate/dense/biasB/bert/encoder/layer_10/intermediate/dense/kernelB+bert/encoder/layer_10/output/LayerNorm/betaB,bert/encoder/layer_10/output/LayerNorm/gammaB'bert/encoder/layer_10/output/dense/biasB)bert/encoder/layer_10/output/dense/kernelB5bert/encoder/layer_11/attention/output/LayerNorm/betaB6bert/encoder/layer_11/attention/output/LayerNorm/gammaB1bert/encoder/layer_11/attention/output/dense/biasB3bert/encoder/layer_11/attention/output/dense/kernelB-bert/encoder/layer_11/attention/self/key/biasB/bert/encoder/layer_11/attention/self/key/kernelB/bert/encoder/layer_11/attention/self/query/biasB1bert/encoder/layer_11/attention/self/query/kernelB/bert/encoder/layer_11/attention/self/value/biasB1bert/encoder/layer_11/attention/self/value/kernelB-bert/encoder/layer_11/intermediate/dense/biasB/bert/encoder/layer_11/intermediate/dense/kernelB+bert/encoder/layer_11/output/LayerNorm/betaB,bert/encoder/layer_11/output/LayerNorm/gammaB'bert/encoder/layer_11/output/dense/biasB)bert/encoder/layer_11/output/dense/kernelB4bert/encoder/layer_2/attention/output/LayerNorm/betaB5bert/encoder/layer_2/attention/output/LayerNorm/gammaB0bert/encoder/layer_2/attention/output/dense/biasB2bert/encoder/layer_2/attention/output/dense/kernelB,bert/encoder/layer_2/attention/self/key/biasB.bert/encoder/layer_2/attention/self/key/kernelB.bert/encoder/layer_2/attention/self/query/biasB0bert/encoder/layer_2/attention/self/query/kernelB.bert/encoder/layer_2/attention/self/value/biasB0bert/encoder/layer_2/attention/self/value/kernelB,bert/encoder/layer_2/intermediate/dense/biasB.bert/encoder/layer_2/intermediate/dense/kernelB*bert/encoder/layer_2/output/LayerNorm/betaB+bert/encoder/layer_2/output/LayerNorm/gammaB&bert/encoder/layer_2/output/dense/biasB(bert/encoder/layer_2/output/dense/kernelB4bert/encoder/layer_3/attention/output/LayerNorm/betaB5bert/encoder/layer_3/attention/output/LayerNorm/gammaB0bert/encoder/layer_3/attention/output/dense/biasB2bert/encoder/layer_3/attention/output/dense/kernelB,bert/encoder/layer_3/attention/self/key/biasB.bert/encoder/layer_3/attention/self/key/kernelB.bert/encoder/layer_3/attention/self/query/biasB0bert/encoder/layer_3/attention/self/query/kernelB.bert/encoder/layer_3/attention/self/value/biasB0bert/encoder/layer_3/attention/self/value/kernelB,bert/encoder/layer_3/intermediate/dense/biasB.bert/encoder/layer_3/intermediate/dense/kernelB*bert/encoder/layer_3/output/LayerNorm/betaB+bert/encoder/layer_3/output/LayerNorm/gammaB&bert/encoder/layer_3/output/dense/biasB(bert/encoder/layer_3/output/dense/kernelB4bert/encoder/layer_4/attention/output/LayerNorm/betaB5bert/encoder/layer_4/attention/output/LayerNorm/gammaB0bert/encoder/layer_4/attention/output/dense/biasB2bert/encoder/layer_4/attention/output/dense/kernelB,bert/encoder/layer_4/attention/self/key/biasB.bert/encoder/layer_4/attention/self/key/kernelB.bert/encoder/layer_4/attention/self/query/biasB0bert/encoder/layer_4/attention/self/query/kernelB.bert/encoder/layer_4/attention/self/value/biasB0bert/encoder/layer_4/attention/self/value/kernelB,bert/encoder/layer_4/intermediate/dense/biasB.bert/encoder/layer_4/intermediate/dense/kernelB*bert/encoder/layer_4/output/LayerNorm/betaB+bert/encoder/layer_4/output/LayerNorm/gammaB&bert/encoder/layer_4/output/dense/biasB(bert/encoder/layer_4/output/dense/kernelB4bert/encoder/layer_5/attention/output/LayerNorm/betaB5bert/encoder/layer_5/attention/output/LayerNorm/gammaB0bert/encoder/layer_5/attention/output/dense/biasB2bert/encoder/layer_5/attention/output/dense/kernelB,bert/encoder/layer_5/attention/self/key/biasB.bert/encoder/layer_5/attention/self/key/kernelB.bert/encoder/layer_5/attention/self/query/biasB0bert/encoder/layer_5/attention/self/query/kernelB.bert/encoder/layer_5/attention/self/value/biasB0bert/encoder/layer_5/attention/self/value/kernelB,bert/encoder/layer_5/intermediate/dense/biasB.bert/encoder/layer_5/intermediate/dense/kernelB*bert/encoder/layer_5/output/LayerNorm/betaB+bert/encoder/layer_5/output/LayerNorm/gammaB&bert/encoder/layer_5/output/dense/biasB(bert/encoder/layer_5/output/dense/kernelB4bert/encoder/layer_6/attention/output/LayerNorm/betaB5bert/encoder/layer_6/attention/output/LayerNorm/gammaB0bert/encoder/layer_6/attention/output/dense/biasB2bert/encoder/layer_6/attention/output/dense/kernelB,bert/encoder/layer_6/attention/self/key/biasB.bert/encoder/layer_6/attention/self/key/kernelB.bert/encoder/layer_6/attention/self/query/biasB0bert/encoder/layer_6/attention/self/query/kernelB.bert/encoder/layer_6/attention/self/value/biasB0bert/encoder/layer_6/attention/self/value/kernelB,bert/encoder/layer_6/intermediate/dense/biasB.bert/encoder/layer_6/intermediate/dense/kernelB*bert/encoder/layer_6/output/LayerNorm/betaB+bert/encoder/layer_6/output/LayerNorm/gammaB&bert/encoder/layer_6/output/dense/biasB(bert/encoder/layer_6/output/dense/kernelB4bert/encoder/layer_7/attention/output/LayerNorm/betaB5bert/encoder/layer_7/attention/output/LayerNorm/gammaB0bert/encoder/layer_7/attention/output/dense/biasB2bert/encoder/layer_7/attention/output/dense/kernelB,bert/encoder/layer_7/attention/self/key/biasB.bert/encoder/layer_7/attention/self/key/kernelB.bert/encoder/layer_7/attention/self/query/biasB0bert/encoder/layer_7/attention/self/query/kernelB.bert/encoder/layer_7/attention/self/value/biasB0bert/encoder/layer_7/attention/self/value/kernelB,bert/encoder/layer_7/intermediate/dense/biasB.bert/encoder/layer_7/intermediate/dense/kernelB*bert/encoder/layer_7/output/LayerNorm/betaB+bert/encoder/layer_7/output/LayerNorm/gammaB&bert/encoder/layer_7/output/dense/biasB(bert/encoder/layer_7/output/dense/kernelB4bert/encoder/layer_8/attention/output/LayerNorm/betaB5bert/encoder/layer_8/attention/output/LayerNorm/gammaB0bert/encoder/layer_8/attention/output/dense/biasB2bert/encoder/layer_8/attention/output/dense/kernelB,bert/encoder/layer_8/attention/self/key/biasB.bert/encoder/layer_8/attention/self/key/kernelB.bert/encoder/layer_8/attention/self/query/biasB0bert/encoder/layer_8/attention/self/query/kernelB.bert/encoder/layer_8/attention/self/value/biasB0bert/encoder/layer_8/attention/self/value/kernelB,bert/encoder/layer_8/intermediate/dense/biasB.bert/encoder/layer_8/intermediate/dense/kernelB*bert/encoder/layer_8/output/LayerNorm/betaB+bert/encoder/layer_8/output/LayerNorm/gammaB&bert/encoder/layer_8/output/dense/biasB(bert/encoder/layer_8/output/dense/kernelB4bert/encoder/layer_9/attention/output/LayerNorm/betaB5bert/encoder/layer_9/attention/output/LayerNorm/gammaB0bert/encoder/layer_9/attention/output/dense/biasB2bert/encoder/layer_9/attention/output/dense/kernelB,bert/encoder/layer_9/attention/self/key/biasB.bert/encoder/layer_9/attention/self/key/kernelB.bert/encoder/layer_9/attention/self/query/biasB0bert/encoder/layer_9/attention/self/query/kernelB.bert/encoder/layer_9/attention/self/value/biasB0bert/encoder/layer_9/attention/self/value/kernelB,bert/encoder/layer_9/intermediate/dense/biasB.bert/encoder/layer_9/intermediate/dense/kernelB*bert/encoder/layer_9/output/LayerNorm/betaB+bert/encoder/layer_9/output/LayerNorm/gammaB&bert/encoder/layer_9/output/dense/biasB(bert/encoder/layer_9/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBoutput_biasBoutput_weights*
dtype0*
_output_shapes	
:?
?
save/SaveV2/shape_and_slicesConst*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:?
?L
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbert/embeddings/LayerNorm/betabert/embeddings/LayerNorm/gamma#bert/embeddings/position_embeddings%bert/embeddings/token_type_embeddingsbert/embeddings/word_embeddings4bert/encoder/layer_0/attention/output/LayerNorm/beta5bert/encoder/layer_0/attention/output/LayerNorm/gamma0bert/encoder/layer_0/attention/output/dense/bias2bert/encoder/layer_0/attention/output/dense/kernel,bert/encoder/layer_0/attention/self/key/bias.bert/encoder/layer_0/attention/self/key/kernel.bert/encoder/layer_0/attention/self/query/bias0bert/encoder/layer_0/attention/self/query/kernel.bert/encoder/layer_0/attention/self/value/bias0bert/encoder/layer_0/attention/self/value/kernel,bert/encoder/layer_0/intermediate/dense/bias.bert/encoder/layer_0/intermediate/dense/kernel*bert/encoder/layer_0/output/LayerNorm/beta+bert/encoder/layer_0/output/LayerNorm/gamma&bert/encoder/layer_0/output/dense/bias(bert/encoder/layer_0/output/dense/kernel4bert/encoder/layer_1/attention/output/LayerNorm/beta5bert/encoder/layer_1/attention/output/LayerNorm/gamma0bert/encoder/layer_1/attention/output/dense/bias2bert/encoder/layer_1/attention/output/dense/kernel,bert/encoder/layer_1/attention/self/key/bias.bert/encoder/layer_1/attention/self/key/kernel.bert/encoder/layer_1/attention/self/query/bias0bert/encoder/layer_1/attention/self/query/kernel.bert/encoder/layer_1/attention/self/value/bias0bert/encoder/layer_1/attention/self/value/kernel,bert/encoder/layer_1/intermediate/dense/bias.bert/encoder/layer_1/intermediate/dense/kernel*bert/encoder/layer_1/output/LayerNorm/beta+bert/encoder/layer_1/output/LayerNorm/gamma&bert/encoder/layer_1/output/dense/bias(bert/encoder/layer_1/output/dense/kernel5bert/encoder/layer_10/attention/output/LayerNorm/beta6bert/encoder/layer_10/attention/output/LayerNorm/gamma1bert/encoder/layer_10/attention/output/dense/bias3bert/encoder/layer_10/attention/output/dense/kernel-bert/encoder/layer_10/attention/self/key/bias/bert/encoder/layer_10/attention/self/key/kernel/bert/encoder/layer_10/attention/self/query/bias1bert/encoder/layer_10/attention/self/query/kernel/bert/encoder/layer_10/attention/self/value/bias1bert/encoder/layer_10/attention/self/value/kernel-bert/encoder/layer_10/intermediate/dense/bias/bert/encoder/layer_10/intermediate/dense/kernel+bert/encoder/layer_10/output/LayerNorm/beta,bert/encoder/layer_10/output/LayerNorm/gamma'bert/encoder/layer_10/output/dense/bias)bert/encoder/layer_10/output/dense/kernel5bert/encoder/layer_11/attention/output/LayerNorm/beta6bert/encoder/layer_11/attention/output/LayerNorm/gamma1bert/encoder/layer_11/attention/output/dense/bias3bert/encoder/layer_11/attention/output/dense/kernel-bert/encoder/layer_11/attention/self/key/bias/bert/encoder/layer_11/attention/self/key/kernel/bert/encoder/layer_11/attention/self/query/bias1bert/encoder/layer_11/attention/self/query/kernel/bert/encoder/layer_11/attention/self/value/bias1bert/encoder/layer_11/attention/self/value/kernel-bert/encoder/layer_11/intermediate/dense/bias/bert/encoder/layer_11/intermediate/dense/kernel+bert/encoder/layer_11/output/LayerNorm/beta,bert/encoder/layer_11/output/LayerNorm/gamma'bert/encoder/layer_11/output/dense/bias)bert/encoder/layer_11/output/dense/kernel4bert/encoder/layer_2/attention/output/LayerNorm/beta5bert/encoder/layer_2/attention/output/LayerNorm/gamma0bert/encoder/layer_2/attention/output/dense/bias2bert/encoder/layer_2/attention/output/dense/kernel,bert/encoder/layer_2/attention/self/key/bias.bert/encoder/layer_2/attention/self/key/kernel.bert/encoder/layer_2/attention/self/query/bias0bert/encoder/layer_2/attention/self/query/kernel.bert/encoder/layer_2/attention/self/value/bias0bert/encoder/layer_2/attention/self/value/kernel,bert/encoder/layer_2/intermediate/dense/bias.bert/encoder/layer_2/intermediate/dense/kernel*bert/encoder/layer_2/output/LayerNorm/beta+bert/encoder/layer_2/output/LayerNorm/gamma&bert/encoder/layer_2/output/dense/bias(bert/encoder/layer_2/output/dense/kernel4bert/encoder/layer_3/attention/output/LayerNorm/beta5bert/encoder/layer_3/attention/output/LayerNorm/gamma0bert/encoder/layer_3/attention/output/dense/bias2bert/encoder/layer_3/attention/output/dense/kernel,bert/encoder/layer_3/attention/self/key/bias.bert/encoder/layer_3/attention/self/key/kernel.bert/encoder/layer_3/attention/self/query/bias0bert/encoder/layer_3/attention/self/query/kernel.bert/encoder/layer_3/attention/self/value/bias0bert/encoder/layer_3/attention/self/value/kernel,bert/encoder/layer_3/intermediate/dense/bias.bert/encoder/layer_3/intermediate/dense/kernel*bert/encoder/layer_3/output/LayerNorm/beta+bert/encoder/layer_3/output/LayerNorm/gamma&bert/encoder/layer_3/output/dense/bias(bert/encoder/layer_3/output/dense/kernel4bert/encoder/layer_4/attention/output/LayerNorm/beta5bert/encoder/layer_4/attention/output/LayerNorm/gamma0bert/encoder/layer_4/attention/output/dense/bias2bert/encoder/layer_4/attention/output/dense/kernel,bert/encoder/layer_4/attention/self/key/bias.bert/encoder/layer_4/attention/self/key/kernel.bert/encoder/layer_4/attention/self/query/bias0bert/encoder/layer_4/attention/self/query/kernel.bert/encoder/layer_4/attention/self/value/bias0bert/encoder/layer_4/attention/self/value/kernel,bert/encoder/layer_4/intermediate/dense/bias.bert/encoder/layer_4/intermediate/dense/kernel*bert/encoder/layer_4/output/LayerNorm/beta+bert/encoder/layer_4/output/LayerNorm/gamma&bert/encoder/layer_4/output/dense/bias(bert/encoder/layer_4/output/dense/kernel4bert/encoder/layer_5/attention/output/LayerNorm/beta5bert/encoder/layer_5/attention/output/LayerNorm/gamma0bert/encoder/layer_5/attention/output/dense/bias2bert/encoder/layer_5/attention/output/dense/kernel,bert/encoder/layer_5/attention/self/key/bias.bert/encoder/layer_5/attention/self/key/kernel.bert/encoder/layer_5/attention/self/query/bias0bert/encoder/layer_5/attention/self/query/kernel.bert/encoder/layer_5/attention/self/value/bias0bert/encoder/layer_5/attention/self/value/kernel,bert/encoder/layer_5/intermediate/dense/bias.bert/encoder/layer_5/intermediate/dense/kernel*bert/encoder/layer_5/output/LayerNorm/beta+bert/encoder/layer_5/output/LayerNorm/gamma&bert/encoder/layer_5/output/dense/bias(bert/encoder/layer_5/output/dense/kernel4bert/encoder/layer_6/attention/output/LayerNorm/beta5bert/encoder/layer_6/attention/output/LayerNorm/gamma0bert/encoder/layer_6/attention/output/dense/bias2bert/encoder/layer_6/attention/output/dense/kernel,bert/encoder/layer_6/attention/self/key/bias.bert/encoder/layer_6/attention/self/key/kernel.bert/encoder/layer_6/attention/self/query/bias0bert/encoder/layer_6/attention/self/query/kernel.bert/encoder/layer_6/attention/self/value/bias0bert/encoder/layer_6/attention/self/value/kernel,bert/encoder/layer_6/intermediate/dense/bias.bert/encoder/layer_6/intermediate/dense/kernel*bert/encoder/layer_6/output/LayerNorm/beta+bert/encoder/layer_6/output/LayerNorm/gamma&bert/encoder/layer_6/output/dense/bias(bert/encoder/layer_6/output/dense/kernel4bert/encoder/layer_7/attention/output/LayerNorm/beta5bert/encoder/layer_7/attention/output/LayerNorm/gamma0bert/encoder/layer_7/attention/output/dense/bias2bert/encoder/layer_7/attention/output/dense/kernel,bert/encoder/layer_7/attention/self/key/bias.bert/encoder/layer_7/attention/self/key/kernel.bert/encoder/layer_7/attention/self/query/bias0bert/encoder/layer_7/attention/self/query/kernel.bert/encoder/layer_7/attention/self/value/bias0bert/encoder/layer_7/attention/self/value/kernel,bert/encoder/layer_7/intermediate/dense/bias.bert/encoder/layer_7/intermediate/dense/kernel*bert/encoder/layer_7/output/LayerNorm/beta+bert/encoder/layer_7/output/LayerNorm/gamma&bert/encoder/layer_7/output/dense/bias(bert/encoder/layer_7/output/dense/kernel4bert/encoder/layer_8/attention/output/LayerNorm/beta5bert/encoder/layer_8/attention/output/LayerNorm/gamma0bert/encoder/layer_8/attention/output/dense/bias2bert/encoder/layer_8/attention/output/dense/kernel,bert/encoder/layer_8/attention/self/key/bias.bert/encoder/layer_8/attention/self/key/kernel.bert/encoder/layer_8/attention/self/query/bias0bert/encoder/layer_8/attention/self/query/kernel.bert/encoder/layer_8/attention/self/value/bias0bert/encoder/layer_8/attention/self/value/kernel,bert/encoder/layer_8/intermediate/dense/bias.bert/encoder/layer_8/intermediate/dense/kernel*bert/encoder/layer_8/output/LayerNorm/beta+bert/encoder/layer_8/output/LayerNorm/gamma&bert/encoder/layer_8/output/dense/bias(bert/encoder/layer_8/output/dense/kernel4bert/encoder/layer_9/attention/output/LayerNorm/beta5bert/encoder/layer_9/attention/output/LayerNorm/gamma0bert/encoder/layer_9/attention/output/dense/bias2bert/encoder/layer_9/attention/output/dense/kernel,bert/encoder/layer_9/attention/self/key/bias.bert/encoder/layer_9/attention/self/key/kernel.bert/encoder/layer_9/attention/self/query/bias0bert/encoder/layer_9/attention/self/query/kernel.bert/encoder/layer_9/attention/self/value/bias0bert/encoder/layer_9/attention/self/value/kernel,bert/encoder/layer_9/intermediate/dense/bias.bert/encoder/layer_9/intermediate/dense/kernel*bert/encoder/layer_9/output/LayerNorm/beta+bert/encoder/layer_9/output/LayerNorm/gamma&bert/encoder/layer_9/output/dense/bias(bert/encoder/layer_9/output/dense/kernelbert/pooler/dense/biasbert/pooler/dense/kerneloutput_biasoutput_weights*?
dtypes?
?2?
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
?J
save/RestoreV2/tensor_namesConst*?J
value?JB?J?Bbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelB5bert/encoder/layer_10/attention/output/LayerNorm/betaB6bert/encoder/layer_10/attention/output/LayerNorm/gammaB1bert/encoder/layer_10/attention/output/dense/biasB3bert/encoder/layer_10/attention/output/dense/kernelB-bert/encoder/layer_10/attention/self/key/biasB/bert/encoder/layer_10/attention/self/key/kernelB/bert/encoder/layer_10/attention/self/query/biasB1bert/encoder/layer_10/attention/self/query/kernelB/bert/encoder/layer_10/attention/self/value/biasB1bert/encoder/layer_10/attention/self/value/kernelB-bert/encoder/layer_10/intermediate/dense/biasB/bert/encoder/layer_10/intermediate/dense/kernelB+bert/encoder/layer_10/output/LayerNorm/betaB,bert/encoder/layer_10/output/LayerNorm/gammaB'bert/encoder/layer_10/output/dense/biasB)bert/encoder/layer_10/output/dense/kernelB5bert/encoder/layer_11/attention/output/LayerNorm/betaB6bert/encoder/layer_11/attention/output/LayerNorm/gammaB1bert/encoder/layer_11/attention/output/dense/biasB3bert/encoder/layer_11/attention/output/dense/kernelB-bert/encoder/layer_11/attention/self/key/biasB/bert/encoder/layer_11/attention/self/key/kernelB/bert/encoder/layer_11/attention/self/query/biasB1bert/encoder/layer_11/attention/self/query/kernelB/bert/encoder/layer_11/attention/self/value/biasB1bert/encoder/layer_11/attention/self/value/kernelB-bert/encoder/layer_11/intermediate/dense/biasB/bert/encoder/layer_11/intermediate/dense/kernelB+bert/encoder/layer_11/output/LayerNorm/betaB,bert/encoder/layer_11/output/LayerNorm/gammaB'bert/encoder/layer_11/output/dense/biasB)bert/encoder/layer_11/output/dense/kernelB4bert/encoder/layer_2/attention/output/LayerNorm/betaB5bert/encoder/layer_2/attention/output/LayerNorm/gammaB0bert/encoder/layer_2/attention/output/dense/biasB2bert/encoder/layer_2/attention/output/dense/kernelB,bert/encoder/layer_2/attention/self/key/biasB.bert/encoder/layer_2/attention/self/key/kernelB.bert/encoder/layer_2/attention/self/query/biasB0bert/encoder/layer_2/attention/self/query/kernelB.bert/encoder/layer_2/attention/self/value/biasB0bert/encoder/layer_2/attention/self/value/kernelB,bert/encoder/layer_2/intermediate/dense/biasB.bert/encoder/layer_2/intermediate/dense/kernelB*bert/encoder/layer_2/output/LayerNorm/betaB+bert/encoder/layer_2/output/LayerNorm/gammaB&bert/encoder/layer_2/output/dense/biasB(bert/encoder/layer_2/output/dense/kernelB4bert/encoder/layer_3/attention/output/LayerNorm/betaB5bert/encoder/layer_3/attention/output/LayerNorm/gammaB0bert/encoder/layer_3/attention/output/dense/biasB2bert/encoder/layer_3/attention/output/dense/kernelB,bert/encoder/layer_3/attention/self/key/biasB.bert/encoder/layer_3/attention/self/key/kernelB.bert/encoder/layer_3/attention/self/query/biasB0bert/encoder/layer_3/attention/self/query/kernelB.bert/encoder/layer_3/attention/self/value/biasB0bert/encoder/layer_3/attention/self/value/kernelB,bert/encoder/layer_3/intermediate/dense/biasB.bert/encoder/layer_3/intermediate/dense/kernelB*bert/encoder/layer_3/output/LayerNorm/betaB+bert/encoder/layer_3/output/LayerNorm/gammaB&bert/encoder/layer_3/output/dense/biasB(bert/encoder/layer_3/output/dense/kernelB4bert/encoder/layer_4/attention/output/LayerNorm/betaB5bert/encoder/layer_4/attention/output/LayerNorm/gammaB0bert/encoder/layer_4/attention/output/dense/biasB2bert/encoder/layer_4/attention/output/dense/kernelB,bert/encoder/layer_4/attention/self/key/biasB.bert/encoder/layer_4/attention/self/key/kernelB.bert/encoder/layer_4/attention/self/query/biasB0bert/encoder/layer_4/attention/self/query/kernelB.bert/encoder/layer_4/attention/self/value/biasB0bert/encoder/layer_4/attention/self/value/kernelB,bert/encoder/layer_4/intermediate/dense/biasB.bert/encoder/layer_4/intermediate/dense/kernelB*bert/encoder/layer_4/output/LayerNorm/betaB+bert/encoder/layer_4/output/LayerNorm/gammaB&bert/encoder/layer_4/output/dense/biasB(bert/encoder/layer_4/output/dense/kernelB4bert/encoder/layer_5/attention/output/LayerNorm/betaB5bert/encoder/layer_5/attention/output/LayerNorm/gammaB0bert/encoder/layer_5/attention/output/dense/biasB2bert/encoder/layer_5/attention/output/dense/kernelB,bert/encoder/layer_5/attention/self/key/biasB.bert/encoder/layer_5/attention/self/key/kernelB.bert/encoder/layer_5/attention/self/query/biasB0bert/encoder/layer_5/attention/self/query/kernelB.bert/encoder/layer_5/attention/self/value/biasB0bert/encoder/layer_5/attention/self/value/kernelB,bert/encoder/layer_5/intermediate/dense/biasB.bert/encoder/layer_5/intermediate/dense/kernelB*bert/encoder/layer_5/output/LayerNorm/betaB+bert/encoder/layer_5/output/LayerNorm/gammaB&bert/encoder/layer_5/output/dense/biasB(bert/encoder/layer_5/output/dense/kernelB4bert/encoder/layer_6/attention/output/LayerNorm/betaB5bert/encoder/layer_6/attention/output/LayerNorm/gammaB0bert/encoder/layer_6/attention/output/dense/biasB2bert/encoder/layer_6/attention/output/dense/kernelB,bert/encoder/layer_6/attention/self/key/biasB.bert/encoder/layer_6/attention/self/key/kernelB.bert/encoder/layer_6/attention/self/query/biasB0bert/encoder/layer_6/attention/self/query/kernelB.bert/encoder/layer_6/attention/self/value/biasB0bert/encoder/layer_6/attention/self/value/kernelB,bert/encoder/layer_6/intermediate/dense/biasB.bert/encoder/layer_6/intermediate/dense/kernelB*bert/encoder/layer_6/output/LayerNorm/betaB+bert/encoder/layer_6/output/LayerNorm/gammaB&bert/encoder/layer_6/output/dense/biasB(bert/encoder/layer_6/output/dense/kernelB4bert/encoder/layer_7/attention/output/LayerNorm/betaB5bert/encoder/layer_7/attention/output/LayerNorm/gammaB0bert/encoder/layer_7/attention/output/dense/biasB2bert/encoder/layer_7/attention/output/dense/kernelB,bert/encoder/layer_7/attention/self/key/biasB.bert/encoder/layer_7/attention/self/key/kernelB.bert/encoder/layer_7/attention/self/query/biasB0bert/encoder/layer_7/attention/self/query/kernelB.bert/encoder/layer_7/attention/self/value/biasB0bert/encoder/layer_7/attention/self/value/kernelB,bert/encoder/layer_7/intermediate/dense/biasB.bert/encoder/layer_7/intermediate/dense/kernelB*bert/encoder/layer_7/output/LayerNorm/betaB+bert/encoder/layer_7/output/LayerNorm/gammaB&bert/encoder/layer_7/output/dense/biasB(bert/encoder/layer_7/output/dense/kernelB4bert/encoder/layer_8/attention/output/LayerNorm/betaB5bert/encoder/layer_8/attention/output/LayerNorm/gammaB0bert/encoder/layer_8/attention/output/dense/biasB2bert/encoder/layer_8/attention/output/dense/kernelB,bert/encoder/layer_8/attention/self/key/biasB.bert/encoder/layer_8/attention/self/key/kernelB.bert/encoder/layer_8/attention/self/query/biasB0bert/encoder/layer_8/attention/self/query/kernelB.bert/encoder/layer_8/attention/self/value/biasB0bert/encoder/layer_8/attention/self/value/kernelB,bert/encoder/layer_8/intermediate/dense/biasB.bert/encoder/layer_8/intermediate/dense/kernelB*bert/encoder/layer_8/output/LayerNorm/betaB+bert/encoder/layer_8/output/LayerNorm/gammaB&bert/encoder/layer_8/output/dense/biasB(bert/encoder/layer_8/output/dense/kernelB4bert/encoder/layer_9/attention/output/LayerNorm/betaB5bert/encoder/layer_9/attention/output/LayerNorm/gammaB0bert/encoder/layer_9/attention/output/dense/biasB2bert/encoder/layer_9/attention/output/dense/kernelB,bert/encoder/layer_9/attention/self/key/biasB.bert/encoder/layer_9/attention/self/key/kernelB.bert/encoder/layer_9/attention/self/query/biasB0bert/encoder/layer_9/attention/self/query/kernelB.bert/encoder/layer_9/attention/self/value/biasB0bert/encoder/layer_9/attention/self/value/kernelB,bert/encoder/layer_9/intermediate/dense/biasB.bert/encoder/layer_9/intermediate/dense/kernelB*bert/encoder/layer_9/output/LayerNorm/betaB+bert/encoder/layer_9/output/LayerNorm/gammaB&bert/encoder/layer_9/output/dense/biasB(bert/encoder/layer_9/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBoutput_biasBoutput_weights*
dtype0*
_output_shapes	
:?
?
save/RestoreV2/shape_and_slicesConst*
_output_shapes	
:?*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*?
dtypes?
?2?*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save/AssignAssignbert/embeddings/LayerNorm/betasave/RestoreV2*
use_locking(*
T0*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_1Assignbert/embeddings/LayerNorm/gammasave/RestoreV2:1*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_2Assign#bert/embeddings/position_embeddingssave/RestoreV2:2*
use_locking(*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_3Assign%bert/embeddings/token_type_embeddingssave/RestoreV2:3*
use_locking(*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
validate_shape(*
_output_shapes
:	?
?
save/Assign_4Assignbert/embeddings/word_embeddingssave/RestoreV2:4*2
_class(
&$loc:@bert/embeddings/word_embeddings*
validate_shape(*!
_output_shapes
:???*
use_locking(*
T0
?
save/Assign_5Assign4bert/encoder/layer_0/attention/output/LayerNorm/betasave/RestoreV2:5*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta
?
save/Assign_6Assign5bert/encoder/layer_0/attention/output/LayerNorm/gammasave/RestoreV2:6*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
validate_shape(
?
save/Assign_7Assign0bert/encoder/layer_0/attention/output/dense/biassave/RestoreV2:7*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_8Assign2bert/encoder/layer_0/attention/output/dense/kernelsave/RestoreV2:8*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel
?
save/Assign_9Assign,bert/encoder/layer_0/attention/self/key/biassave/RestoreV2:9*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
validate_shape(
?
save/Assign_10Assign.bert/encoder/layer_0/attention/self/key/kernelsave/RestoreV2:10*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_11Assign.bert/encoder/layer_0/attention/self/query/biassave/RestoreV2:11*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_12Assign0bert/encoder/layer_0/attention/self/query/kernelsave/RestoreV2:12*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_13Assign.bert/encoder/layer_0/attention/self/value/biassave/RestoreV2:13*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
validate_shape(
?
save/Assign_14Assign0bert/encoder/layer_0/attention/self/value/kernelsave/RestoreV2:14*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save/Assign_15Assign,bert/encoder/layer_0/intermediate/dense/biassave/RestoreV2:15*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
validate_shape(
?
save/Assign_16Assign.bert/encoder/layer_0/intermediate/dense/kernelsave/RestoreV2:16*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_17Assign*bert/encoder/layer_0/output/LayerNorm/betasave/RestoreV2:17*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
validate_shape(
?
save/Assign_18Assign+bert/encoder/layer_0/output/LayerNorm/gammasave/RestoreV2:18*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_19Assign&bert/encoder/layer_0/output/dense/biassave/RestoreV2:19*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias
?
save/Assign_20Assign(bert/encoder/layer_0/output/dense/kernelsave/RestoreV2:20*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_21Assign4bert/encoder/layer_1/attention/output/LayerNorm/betasave/RestoreV2:21*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_22Assign5bert/encoder/layer_1/attention/output/LayerNorm/gammasave/RestoreV2:22*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_23Assign0bert/encoder/layer_1/attention/output/dense/biassave/RestoreV2:23*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
validate_shape(
?
save/Assign_24Assign2bert/encoder/layer_1/attention/output/dense/kernelsave/RestoreV2:24*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_25Assign,bert/encoder/layer_1/attention/self/key/biassave/RestoreV2:25*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_26Assign.bert/encoder/layer_1/attention/self/key/kernelsave/RestoreV2:26* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
validate_shape(
?
save/Assign_27Assign.bert/encoder/layer_1/attention/self/query/biassave/RestoreV2:27*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_28Assign0bert/encoder/layer_1/attention/self/query/kernelsave/RestoreV2:28* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
validate_shape(
?
save/Assign_29Assign.bert/encoder/layer_1/attention/self/value/biassave/RestoreV2:29*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_30Assign0bert/encoder/layer_1/attention/self/value/kernelsave/RestoreV2:30* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
validate_shape(
?
save/Assign_31Assign,bert/encoder/layer_1/intermediate/dense/biassave/RestoreV2:31*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_32Assign.bert/encoder/layer_1/intermediate/dense/kernelsave/RestoreV2:32* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
validate_shape(
?
save/Assign_33Assign*bert/encoder/layer_1/output/LayerNorm/betasave/RestoreV2:33*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
validate_shape(
?
save/Assign_34Assign+bert/encoder/layer_1/output/LayerNorm/gammasave/RestoreV2:34*
T0*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_35Assign&bert/encoder/layer_1/output/dense/biassave/RestoreV2:35*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_36Assign(bert/encoder/layer_1/output/dense/kernelsave/RestoreV2:36*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_37Assign5bert/encoder/layer_10/attention/output/LayerNorm/betasave/RestoreV2:37*H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_38Assign6bert/encoder/layer_10/attention/output/LayerNorm/gammasave/RestoreV2:38*
use_locking(*
T0*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_39Assign1bert/encoder/layer_10/attention/output/dense/biassave/RestoreV2:39*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias
?
save/Assign_40Assign3bert/encoder/layer_10/attention/output/dense/kernelsave/RestoreV2:40* 
_output_shapes
:
??*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
validate_shape(
?
save/Assign_41Assign-bert/encoder/layer_10/attention/self/key/biassave/RestoreV2:41*
_output_shapes	
:?*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
validate_shape(
?
save/Assign_42Assign/bert/encoder/layer_10/attention/self/key/kernelsave/RestoreV2:42*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save/Assign_43Assign/bert/encoder/layer_10/attention/self/query/biassave/RestoreV2:43*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_44Assign1bert/encoder/layer_10/attention/self/query/kernelsave/RestoreV2:44*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_45Assign/bert/encoder/layer_10/attention/self/value/biassave/RestoreV2:45*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias
?
save/Assign_46Assign1bert/encoder/layer_10/attention/self/value/kernelsave/RestoreV2:46*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_47Assign-bert/encoder/layer_10/intermediate/dense/biassave/RestoreV2:47*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias
?
save/Assign_48Assign/bert/encoder/layer_10/intermediate/dense/kernelsave/RestoreV2:48*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_49Assign+bert/encoder/layer_10/output/LayerNorm/betasave/RestoreV2:49*
_output_shapes	
:?*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
validate_shape(
?
save/Assign_50Assign,bert/encoder/layer_10/output/LayerNorm/gammasave/RestoreV2:50*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
validate_shape(
?
save/Assign_51Assign'bert/encoder/layer_10/output/dense/biassave/RestoreV2:51*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias
?
save/Assign_52Assign)bert/encoder/layer_10/output/dense/kernelsave/RestoreV2:52*
use_locking(*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_53Assign5bert/encoder/layer_11/attention/output/LayerNorm/betasave/RestoreV2:53*H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_54Assign6bert/encoder/layer_11/attention/output/LayerNorm/gammasave/RestoreV2:54*
use_locking(*
T0*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_55Assign1bert/encoder/layer_11/attention/output/dense/biassave/RestoreV2:55*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_56Assign3bert/encoder/layer_11/attention/output/dense/kernelsave/RestoreV2:56*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_57Assign-bert/encoder/layer_11/attention/self/key/biassave/RestoreV2:57*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias
?
save/Assign_58Assign/bert/encoder/layer_11/attention/self/key/kernelsave/RestoreV2:58*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_59Assign/bert/encoder/layer_11/attention/self/query/biassave/RestoreV2:59*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_60Assign1bert/encoder/layer_11/attention/self/query/kernelsave/RestoreV2:60*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel
?
save/Assign_61Assign/bert/encoder/layer_11/attention/self/value/biassave/RestoreV2:61*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_62Assign1bert/encoder/layer_11/attention/self/value/kernelsave/RestoreV2:62*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_63Assign-bert/encoder/layer_11/intermediate/dense/biassave/RestoreV2:63*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_64Assign/bert/encoder/layer_11/intermediate/dense/kernelsave/RestoreV2:64*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_65Assign+bert/encoder/layer_11/output/LayerNorm/betasave/RestoreV2:65*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_66Assign,bert/encoder/layer_11/output/LayerNorm/gammasave/RestoreV2:66*
T0*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_67Assign'bert/encoder/layer_11/output/dense/biassave/RestoreV2:67*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_68Assign)bert/encoder/layer_11/output/dense/kernelsave/RestoreV2:68*
use_locking(*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_69Assign4bert/encoder/layer_2/attention/output/LayerNorm/betasave/RestoreV2:69*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_70Assign5bert/encoder/layer_2/attention/output/LayerNorm/gammasave/RestoreV2:70*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_71Assign0bert/encoder/layer_2/attention/output/dense/biassave/RestoreV2:71*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
validate_shape(
?
save/Assign_72Assign2bert/encoder/layer_2/attention/output/dense/kernelsave/RestoreV2:72*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel
?
save/Assign_73Assign,bert/encoder/layer_2/attention/self/key/biassave/RestoreV2:73*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_74Assign.bert/encoder/layer_2/attention/self/key/kernelsave/RestoreV2:74*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_75Assign.bert/encoder/layer_2/attention/self/query/biassave/RestoreV2:75*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
validate_shape(
?
save/Assign_76Assign0bert/encoder/layer_2/attention/self/query/kernelsave/RestoreV2:76*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_77Assign.bert/encoder/layer_2/attention/self/value/biassave/RestoreV2:77*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_78Assign0bert/encoder/layer_2/attention/self/value/kernelsave/RestoreV2:78* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
validate_shape(
?
save/Assign_79Assign,bert/encoder/layer_2/intermediate/dense/biassave/RestoreV2:79*
T0*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_80Assign.bert/encoder/layer_2/intermediate/dense/kernelsave/RestoreV2:80*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_81Assign*bert/encoder/layer_2/output/LayerNorm/betasave/RestoreV2:81*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta
?
save/Assign_82Assign+bert/encoder/layer_2/output/LayerNorm/gammasave/RestoreV2:82*
T0*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_83Assign&bert/encoder/layer_2/output/dense/biassave/RestoreV2:83*
T0*9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_84Assign(bert/encoder/layer_2/output/dense/kernelsave/RestoreV2:84* 
_output_shapes
:
??*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
validate_shape(
?
save/Assign_85Assign4bert/encoder/layer_3/attention/output/LayerNorm/betasave/RestoreV2:85*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_86Assign5bert/encoder/layer_3/attention/output/LayerNorm/gammasave/RestoreV2:86*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_87Assign0bert/encoder/layer_3/attention/output/dense/biassave/RestoreV2:87*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_88Assign2bert/encoder/layer_3/attention/output/dense/kernelsave/RestoreV2:88*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_89Assign,bert/encoder/layer_3/attention/self/key/biassave/RestoreV2:89*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_90Assign.bert/encoder/layer_3/attention/self/key/kernelsave/RestoreV2:90*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_91Assign.bert/encoder/layer_3/attention/self/query/biassave/RestoreV2:91*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_92Assign0bert/encoder/layer_3/attention/self/query/kernelsave/RestoreV2:92*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_93Assign.bert/encoder/layer_3/attention/self/value/biassave/RestoreV2:93*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_94Assign0bert/encoder/layer_3/attention/self/value/kernelsave/RestoreV2:94*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_95Assign,bert/encoder/layer_3/intermediate/dense/biassave/RestoreV2:95*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_96Assign.bert/encoder/layer_3/intermediate/dense/kernelsave/RestoreV2:96* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
validate_shape(
?
save/Assign_97Assign*bert/encoder/layer_3/output/LayerNorm/betasave/RestoreV2:97*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta
?
save/Assign_98Assign+bert/encoder/layer_3/output/LayerNorm/gammasave/RestoreV2:98*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma
?
save/Assign_99Assign&bert/encoder/layer_3/output/dense/biassave/RestoreV2:99*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_100Assign(bert/encoder/layer_3/output/dense/kernelsave/RestoreV2:100*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_101Assign4bert/encoder/layer_4/attention/output/LayerNorm/betasave/RestoreV2:101*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_102Assign5bert/encoder/layer_4/attention/output/LayerNorm/gammasave/RestoreV2:102*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_103Assign0bert/encoder/layer_4/attention/output/dense/biassave/RestoreV2:103*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_104Assign2bert/encoder/layer_4/attention/output/dense/kernelsave/RestoreV2:104*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_105Assign,bert/encoder/layer_4/attention/self/key/biassave/RestoreV2:105*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_106Assign.bert/encoder/layer_4/attention/self/key/kernelsave/RestoreV2:106*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_107Assign.bert/encoder/layer_4/attention/self/query/biassave/RestoreV2:107*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_108Assign0bert/encoder/layer_4/attention/self/query/kernelsave/RestoreV2:108* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
validate_shape(
?
save/Assign_109Assign.bert/encoder/layer_4/attention/self/value/biassave/RestoreV2:109*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_110Assign0bert/encoder/layer_4/attention/self/value/kernelsave/RestoreV2:110*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel
?
save/Assign_111Assign,bert/encoder/layer_4/intermediate/dense/biassave/RestoreV2:111*
T0*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_112Assign.bert/encoder/layer_4/intermediate/dense/kernelsave/RestoreV2:112*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_113Assign*bert/encoder/layer_4/output/LayerNorm/betasave/RestoreV2:113*
T0*=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_114Assign+bert/encoder/layer_4/output/LayerNorm/gammasave/RestoreV2:114*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_115Assign&bert/encoder/layer_4/output/dense/biassave/RestoreV2:115*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_116Assign(bert/encoder/layer_4/output/dense/kernelsave/RestoreV2:116*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_117Assign4bert/encoder/layer_5/attention/output/LayerNorm/betasave/RestoreV2:117*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta
?
save/Assign_118Assign5bert/encoder/layer_5/attention/output/LayerNorm/gammasave/RestoreV2:118*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_119Assign0bert/encoder/layer_5/attention/output/dense/biassave/RestoreV2:119*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_120Assign2bert/encoder/layer_5/attention/output/dense/kernelsave/RestoreV2:120*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_121Assign,bert/encoder/layer_5/attention/self/key/biassave/RestoreV2:121*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_122Assign.bert/encoder/layer_5/attention/self/key/kernelsave/RestoreV2:122*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save/Assign_123Assign.bert/encoder/layer_5/attention/self/query/biassave/RestoreV2:123*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_124Assign0bert/encoder/layer_5/attention/self/query/kernelsave/RestoreV2:124*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save/Assign_125Assign.bert/encoder/layer_5/attention/self/value/biassave/RestoreV2:125*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias
?
save/Assign_126Assign0bert/encoder/layer_5/attention/self/value/kernelsave/RestoreV2:126*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save/Assign_127Assign,bert/encoder/layer_5/intermediate/dense/biassave/RestoreV2:127*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_128Assign.bert/encoder/layer_5/intermediate/dense/kernelsave/RestoreV2:128*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_129Assign*bert/encoder/layer_5/output/LayerNorm/betasave/RestoreV2:129*
T0*=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_130Assign+bert/encoder/layer_5/output/LayerNorm/gammasave/RestoreV2:130*
T0*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_131Assign&bert/encoder/layer_5/output/dense/biassave/RestoreV2:131*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias
?
save/Assign_132Assign(bert/encoder/layer_5/output/dense/kernelsave/RestoreV2:132* 
_output_shapes
:
??*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
validate_shape(
?
save/Assign_133Assign4bert/encoder/layer_6/attention/output/LayerNorm/betasave/RestoreV2:133*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_134Assign5bert/encoder/layer_6/attention/output/LayerNorm/gammasave/RestoreV2:134*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma
?
save/Assign_135Assign0bert/encoder/layer_6/attention/output/dense/biassave/RestoreV2:135*C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_136Assign2bert/encoder/layer_6/attention/output/dense/kernelsave/RestoreV2:136* 
_output_shapes
:
??*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
validate_shape(
?
save/Assign_137Assign,bert/encoder/layer_6/attention/self/key/biassave/RestoreV2:137*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_138Assign.bert/encoder/layer_6/attention/self/key/kernelsave/RestoreV2:138*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel
?
save/Assign_139Assign.bert/encoder/layer_6/attention/self/query/biassave/RestoreV2:139*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_140Assign0bert/encoder/layer_6/attention/self/query/kernelsave/RestoreV2:140*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_141Assign.bert/encoder/layer_6/attention/self/value/biassave/RestoreV2:141*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_142Assign0bert/encoder/layer_6/attention/self/value/kernelsave/RestoreV2:142*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_143Assign,bert/encoder/layer_6/intermediate/dense/biassave/RestoreV2:143*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias
?
save/Assign_144Assign.bert/encoder/layer_6/intermediate/dense/kernelsave/RestoreV2:144*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_145Assign*bert/encoder/layer_6/output/LayerNorm/betasave/RestoreV2:145*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
validate_shape(
?
save/Assign_146Assign+bert/encoder/layer_6/output/LayerNorm/gammasave/RestoreV2:146*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_147Assign&bert/encoder/layer_6/output/dense/biassave/RestoreV2:147*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_148Assign(bert/encoder/layer_6/output/dense/kernelsave/RestoreV2:148*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_149Assign4bert/encoder/layer_7/attention/output/LayerNorm/betasave/RestoreV2:149*G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_150Assign5bert/encoder/layer_7/attention/output/LayerNorm/gammasave/RestoreV2:150*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
validate_shape(
?
save/Assign_151Assign0bert/encoder/layer_7/attention/output/dense/biassave/RestoreV2:151*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_152Assign2bert/encoder/layer_7/attention/output/dense/kernelsave/RestoreV2:152*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_153Assign,bert/encoder/layer_7/attention/self/key/biassave/RestoreV2:153*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_154Assign.bert/encoder/layer_7/attention/self/key/kernelsave/RestoreV2:154*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_155Assign.bert/encoder/layer_7/attention/self/query/biassave/RestoreV2:155*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_156Assign0bert/encoder/layer_7/attention/self/query/kernelsave/RestoreV2:156*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_157Assign.bert/encoder/layer_7/attention/self/value/biassave/RestoreV2:157*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_158Assign0bert/encoder/layer_7/attention/self/value/kernelsave/RestoreV2:158*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_159Assign,bert/encoder/layer_7/intermediate/dense/biassave/RestoreV2:159*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_160Assign.bert/encoder/layer_7/intermediate/dense/kernelsave/RestoreV2:160*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_161Assign*bert/encoder/layer_7/output/LayerNorm/betasave/RestoreV2:161*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_162Assign+bert/encoder/layer_7/output/LayerNorm/gammasave/RestoreV2:162*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma
?
save/Assign_163Assign&bert/encoder/layer_7/output/dense/biassave/RestoreV2:163*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_164Assign(bert/encoder/layer_7/output/dense/kernelsave/RestoreV2:164*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save/Assign_165Assign4bert/encoder/layer_8/attention/output/LayerNorm/betasave/RestoreV2:165*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_166Assign5bert/encoder/layer_8/attention/output/LayerNorm/gammasave/RestoreV2:166*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_167Assign0bert/encoder/layer_8/attention/output/dense/biassave/RestoreV2:167*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_168Assign2bert/encoder/layer_8/attention/output/dense/kernelsave/RestoreV2:168*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_169Assign,bert/encoder/layer_8/attention/self/key/biassave/RestoreV2:169*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_170Assign.bert/encoder/layer_8/attention/self/key/kernelsave/RestoreV2:170*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_171Assign.bert/encoder/layer_8/attention/self/query/biassave/RestoreV2:171*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_172Assign0bert/encoder/layer_8/attention/self/query/kernelsave/RestoreV2:172* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
validate_shape(
?
save/Assign_173Assign.bert/encoder/layer_8/attention/self/value/biassave/RestoreV2:173*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_174Assign0bert/encoder/layer_8/attention/self/value/kernelsave/RestoreV2:174*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_175Assign,bert/encoder/layer_8/intermediate/dense/biassave/RestoreV2:175*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_176Assign.bert/encoder/layer_8/intermediate/dense/kernelsave/RestoreV2:176*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel
?
save/Assign_177Assign*bert/encoder/layer_8/output/LayerNorm/betasave/RestoreV2:177*
T0*=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_178Assign+bert/encoder/layer_8/output/LayerNorm/gammasave/RestoreV2:178*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma
?
save/Assign_179Assign&bert/encoder/layer_8/output/dense/biassave/RestoreV2:179*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias
?
save/Assign_180Assign(bert/encoder/layer_8/output/dense/kernelsave/RestoreV2:180*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_181Assign4bert/encoder/layer_9/attention/output/LayerNorm/betasave/RestoreV2:181*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta
?
save/Assign_182Assign5bert/encoder/layer_9/attention/output/LayerNorm/gammasave/RestoreV2:182*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_183Assign0bert/encoder/layer_9/attention/output/dense/biassave/RestoreV2:183*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias
?
save/Assign_184Assign2bert/encoder/layer_9/attention/output/dense/kernelsave/RestoreV2:184*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save/Assign_185Assign,bert/encoder/layer_9/attention/self/key/biassave/RestoreV2:185*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
validate_shape(
?
save/Assign_186Assign.bert/encoder/layer_9/attention/self/key/kernelsave/RestoreV2:186*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel
?
save/Assign_187Assign.bert/encoder/layer_9/attention/self/query/biassave/RestoreV2:187*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_188Assign0bert/encoder/layer_9/attention/self/query/kernelsave/RestoreV2:188*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_189Assign.bert/encoder/layer_9/attention/self/value/biassave/RestoreV2:189*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_190Assign0bert/encoder/layer_9/attention/self/value/kernelsave/RestoreV2:190*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_191Assign,bert/encoder/layer_9/intermediate/dense/biassave/RestoreV2:191*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_192Assign.bert/encoder/layer_9/intermediate/dense/kernelsave/RestoreV2:192*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_193Assign*bert/encoder/layer_9/output/LayerNorm/betasave/RestoreV2:193*
T0*=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_194Assign+bert/encoder/layer_9/output/LayerNorm/gammasave/RestoreV2:194*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_195Assign&bert/encoder/layer_9/output/dense/biassave/RestoreV2:195*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_196Assign(bert/encoder/layer_9/output/dense/kernelsave/RestoreV2:196*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save/Assign_197Assignbert/pooler/dense/biassave/RestoreV2:197*
use_locking(*
T0*)
_class
loc:@bert/pooler/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save/Assign_198Assignbert/pooler/dense/kernelsave/RestoreV2:198*
use_locking(*
T0*+
_class!
loc:@bert/pooler/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save/Assign_199Assignoutput_biassave/RestoreV2:199*
_class
loc:@output_bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
?
save/Assign_200Assignoutput_weightssave/RestoreV2:200*
use_locking(*
T0*!
_class
loc:@output_weights*
validate_shape(*
_output_shapes
:	?
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_11^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_12^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_129^save/Assign_13^save/Assign_130^save/Assign_131^save/Assign_132^save/Assign_133^save/Assign_134^save/Assign_135^save/Assign_136^save/Assign_137^save/Assign_138^save/Assign_139^save/Assign_14^save/Assign_140^save/Assign_141^save/Assign_142^save/Assign_143^save/Assign_144^save/Assign_145^save/Assign_146^save/Assign_147^save/Assign_148^save/Assign_149^save/Assign_15^save/Assign_150^save/Assign_151^save/Assign_152^save/Assign_153^save/Assign_154^save/Assign_155^save/Assign_156^save/Assign_157^save/Assign_158^save/Assign_159^save/Assign_16^save/Assign_160^save/Assign_161^save/Assign_162^save/Assign_163^save/Assign_164^save/Assign_165^save/Assign_166^save/Assign_167^save/Assign_168^save/Assign_169^save/Assign_17^save/Assign_170^save/Assign_171^save/Assign_172^save/Assign_173^save/Assign_174^save/Assign_175^save/Assign_176^save/Assign_177^save/Assign_178^save/Assign_179^save/Assign_18^save/Assign_180^save/Assign_181^save/Assign_182^save/Assign_183^save/Assign_184^save/Assign_185^save/Assign_186^save/Assign_187^save/Assign_188^save/Assign_189^save/Assign_19^save/Assign_190^save/Assign_191^save/Assign_192^save/Assign_193^save/Assign_194^save/Assign_195^save/Assign_196^save/Assign_197^save/Assign_198^save/Assign_199^save/Assign_2^save/Assign_20^save/Assign_200^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
?
save_1/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_ecc253766eec47ce948d614f588fdbd3/part*
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
?J
save_1/SaveV2/tensor_namesConst*?J
value?JB?J?Bbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelB5bert/encoder/layer_10/attention/output/LayerNorm/betaB6bert/encoder/layer_10/attention/output/LayerNorm/gammaB1bert/encoder/layer_10/attention/output/dense/biasB3bert/encoder/layer_10/attention/output/dense/kernelB-bert/encoder/layer_10/attention/self/key/biasB/bert/encoder/layer_10/attention/self/key/kernelB/bert/encoder/layer_10/attention/self/query/biasB1bert/encoder/layer_10/attention/self/query/kernelB/bert/encoder/layer_10/attention/self/value/biasB1bert/encoder/layer_10/attention/self/value/kernelB-bert/encoder/layer_10/intermediate/dense/biasB/bert/encoder/layer_10/intermediate/dense/kernelB+bert/encoder/layer_10/output/LayerNorm/betaB,bert/encoder/layer_10/output/LayerNorm/gammaB'bert/encoder/layer_10/output/dense/biasB)bert/encoder/layer_10/output/dense/kernelB5bert/encoder/layer_11/attention/output/LayerNorm/betaB6bert/encoder/layer_11/attention/output/LayerNorm/gammaB1bert/encoder/layer_11/attention/output/dense/biasB3bert/encoder/layer_11/attention/output/dense/kernelB-bert/encoder/layer_11/attention/self/key/biasB/bert/encoder/layer_11/attention/self/key/kernelB/bert/encoder/layer_11/attention/self/query/biasB1bert/encoder/layer_11/attention/self/query/kernelB/bert/encoder/layer_11/attention/self/value/biasB1bert/encoder/layer_11/attention/self/value/kernelB-bert/encoder/layer_11/intermediate/dense/biasB/bert/encoder/layer_11/intermediate/dense/kernelB+bert/encoder/layer_11/output/LayerNorm/betaB,bert/encoder/layer_11/output/LayerNorm/gammaB'bert/encoder/layer_11/output/dense/biasB)bert/encoder/layer_11/output/dense/kernelB4bert/encoder/layer_2/attention/output/LayerNorm/betaB5bert/encoder/layer_2/attention/output/LayerNorm/gammaB0bert/encoder/layer_2/attention/output/dense/biasB2bert/encoder/layer_2/attention/output/dense/kernelB,bert/encoder/layer_2/attention/self/key/biasB.bert/encoder/layer_2/attention/self/key/kernelB.bert/encoder/layer_2/attention/self/query/biasB0bert/encoder/layer_2/attention/self/query/kernelB.bert/encoder/layer_2/attention/self/value/biasB0bert/encoder/layer_2/attention/self/value/kernelB,bert/encoder/layer_2/intermediate/dense/biasB.bert/encoder/layer_2/intermediate/dense/kernelB*bert/encoder/layer_2/output/LayerNorm/betaB+bert/encoder/layer_2/output/LayerNorm/gammaB&bert/encoder/layer_2/output/dense/biasB(bert/encoder/layer_2/output/dense/kernelB4bert/encoder/layer_3/attention/output/LayerNorm/betaB5bert/encoder/layer_3/attention/output/LayerNorm/gammaB0bert/encoder/layer_3/attention/output/dense/biasB2bert/encoder/layer_3/attention/output/dense/kernelB,bert/encoder/layer_3/attention/self/key/biasB.bert/encoder/layer_3/attention/self/key/kernelB.bert/encoder/layer_3/attention/self/query/biasB0bert/encoder/layer_3/attention/self/query/kernelB.bert/encoder/layer_3/attention/self/value/biasB0bert/encoder/layer_3/attention/self/value/kernelB,bert/encoder/layer_3/intermediate/dense/biasB.bert/encoder/layer_3/intermediate/dense/kernelB*bert/encoder/layer_3/output/LayerNorm/betaB+bert/encoder/layer_3/output/LayerNorm/gammaB&bert/encoder/layer_3/output/dense/biasB(bert/encoder/layer_3/output/dense/kernelB4bert/encoder/layer_4/attention/output/LayerNorm/betaB5bert/encoder/layer_4/attention/output/LayerNorm/gammaB0bert/encoder/layer_4/attention/output/dense/biasB2bert/encoder/layer_4/attention/output/dense/kernelB,bert/encoder/layer_4/attention/self/key/biasB.bert/encoder/layer_4/attention/self/key/kernelB.bert/encoder/layer_4/attention/self/query/biasB0bert/encoder/layer_4/attention/self/query/kernelB.bert/encoder/layer_4/attention/self/value/biasB0bert/encoder/layer_4/attention/self/value/kernelB,bert/encoder/layer_4/intermediate/dense/biasB.bert/encoder/layer_4/intermediate/dense/kernelB*bert/encoder/layer_4/output/LayerNorm/betaB+bert/encoder/layer_4/output/LayerNorm/gammaB&bert/encoder/layer_4/output/dense/biasB(bert/encoder/layer_4/output/dense/kernelB4bert/encoder/layer_5/attention/output/LayerNorm/betaB5bert/encoder/layer_5/attention/output/LayerNorm/gammaB0bert/encoder/layer_5/attention/output/dense/biasB2bert/encoder/layer_5/attention/output/dense/kernelB,bert/encoder/layer_5/attention/self/key/biasB.bert/encoder/layer_5/attention/self/key/kernelB.bert/encoder/layer_5/attention/self/query/biasB0bert/encoder/layer_5/attention/self/query/kernelB.bert/encoder/layer_5/attention/self/value/biasB0bert/encoder/layer_5/attention/self/value/kernelB,bert/encoder/layer_5/intermediate/dense/biasB.bert/encoder/layer_5/intermediate/dense/kernelB*bert/encoder/layer_5/output/LayerNorm/betaB+bert/encoder/layer_5/output/LayerNorm/gammaB&bert/encoder/layer_5/output/dense/biasB(bert/encoder/layer_5/output/dense/kernelB4bert/encoder/layer_6/attention/output/LayerNorm/betaB5bert/encoder/layer_6/attention/output/LayerNorm/gammaB0bert/encoder/layer_6/attention/output/dense/biasB2bert/encoder/layer_6/attention/output/dense/kernelB,bert/encoder/layer_6/attention/self/key/biasB.bert/encoder/layer_6/attention/self/key/kernelB.bert/encoder/layer_6/attention/self/query/biasB0bert/encoder/layer_6/attention/self/query/kernelB.bert/encoder/layer_6/attention/self/value/biasB0bert/encoder/layer_6/attention/self/value/kernelB,bert/encoder/layer_6/intermediate/dense/biasB.bert/encoder/layer_6/intermediate/dense/kernelB*bert/encoder/layer_6/output/LayerNorm/betaB+bert/encoder/layer_6/output/LayerNorm/gammaB&bert/encoder/layer_6/output/dense/biasB(bert/encoder/layer_6/output/dense/kernelB4bert/encoder/layer_7/attention/output/LayerNorm/betaB5bert/encoder/layer_7/attention/output/LayerNorm/gammaB0bert/encoder/layer_7/attention/output/dense/biasB2bert/encoder/layer_7/attention/output/dense/kernelB,bert/encoder/layer_7/attention/self/key/biasB.bert/encoder/layer_7/attention/self/key/kernelB.bert/encoder/layer_7/attention/self/query/biasB0bert/encoder/layer_7/attention/self/query/kernelB.bert/encoder/layer_7/attention/self/value/biasB0bert/encoder/layer_7/attention/self/value/kernelB,bert/encoder/layer_7/intermediate/dense/biasB.bert/encoder/layer_7/intermediate/dense/kernelB*bert/encoder/layer_7/output/LayerNorm/betaB+bert/encoder/layer_7/output/LayerNorm/gammaB&bert/encoder/layer_7/output/dense/biasB(bert/encoder/layer_7/output/dense/kernelB4bert/encoder/layer_8/attention/output/LayerNorm/betaB5bert/encoder/layer_8/attention/output/LayerNorm/gammaB0bert/encoder/layer_8/attention/output/dense/biasB2bert/encoder/layer_8/attention/output/dense/kernelB,bert/encoder/layer_8/attention/self/key/biasB.bert/encoder/layer_8/attention/self/key/kernelB.bert/encoder/layer_8/attention/self/query/biasB0bert/encoder/layer_8/attention/self/query/kernelB.bert/encoder/layer_8/attention/self/value/biasB0bert/encoder/layer_8/attention/self/value/kernelB,bert/encoder/layer_8/intermediate/dense/biasB.bert/encoder/layer_8/intermediate/dense/kernelB*bert/encoder/layer_8/output/LayerNorm/betaB+bert/encoder/layer_8/output/LayerNorm/gammaB&bert/encoder/layer_8/output/dense/biasB(bert/encoder/layer_8/output/dense/kernelB4bert/encoder/layer_9/attention/output/LayerNorm/betaB5bert/encoder/layer_9/attention/output/LayerNorm/gammaB0bert/encoder/layer_9/attention/output/dense/biasB2bert/encoder/layer_9/attention/output/dense/kernelB,bert/encoder/layer_9/attention/self/key/biasB.bert/encoder/layer_9/attention/self/key/kernelB.bert/encoder/layer_9/attention/self/query/biasB0bert/encoder/layer_9/attention/self/query/kernelB.bert/encoder/layer_9/attention/self/value/biasB0bert/encoder/layer_9/attention/self/value/kernelB,bert/encoder/layer_9/intermediate/dense/biasB.bert/encoder/layer_9/intermediate/dense/kernelB*bert/encoder/layer_9/output/LayerNorm/betaB+bert/encoder/layer_9/output/LayerNorm/gammaB&bert/encoder/layer_9/output/dense/biasB(bert/encoder/layer_9/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBoutput_biasBoutput_weights*
dtype0*
_output_shapes	
:?
?
save_1/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes	
:?*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?L
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbert/embeddings/LayerNorm/betabert/embeddings/LayerNorm/gamma#bert/embeddings/position_embeddings%bert/embeddings/token_type_embeddingsbert/embeddings/word_embeddings4bert/encoder/layer_0/attention/output/LayerNorm/beta5bert/encoder/layer_0/attention/output/LayerNorm/gamma0bert/encoder/layer_0/attention/output/dense/bias2bert/encoder/layer_0/attention/output/dense/kernel,bert/encoder/layer_0/attention/self/key/bias.bert/encoder/layer_0/attention/self/key/kernel.bert/encoder/layer_0/attention/self/query/bias0bert/encoder/layer_0/attention/self/query/kernel.bert/encoder/layer_0/attention/self/value/bias0bert/encoder/layer_0/attention/self/value/kernel,bert/encoder/layer_0/intermediate/dense/bias.bert/encoder/layer_0/intermediate/dense/kernel*bert/encoder/layer_0/output/LayerNorm/beta+bert/encoder/layer_0/output/LayerNorm/gamma&bert/encoder/layer_0/output/dense/bias(bert/encoder/layer_0/output/dense/kernel4bert/encoder/layer_1/attention/output/LayerNorm/beta5bert/encoder/layer_1/attention/output/LayerNorm/gamma0bert/encoder/layer_1/attention/output/dense/bias2bert/encoder/layer_1/attention/output/dense/kernel,bert/encoder/layer_1/attention/self/key/bias.bert/encoder/layer_1/attention/self/key/kernel.bert/encoder/layer_1/attention/self/query/bias0bert/encoder/layer_1/attention/self/query/kernel.bert/encoder/layer_1/attention/self/value/bias0bert/encoder/layer_1/attention/self/value/kernel,bert/encoder/layer_1/intermediate/dense/bias.bert/encoder/layer_1/intermediate/dense/kernel*bert/encoder/layer_1/output/LayerNorm/beta+bert/encoder/layer_1/output/LayerNorm/gamma&bert/encoder/layer_1/output/dense/bias(bert/encoder/layer_1/output/dense/kernel5bert/encoder/layer_10/attention/output/LayerNorm/beta6bert/encoder/layer_10/attention/output/LayerNorm/gamma1bert/encoder/layer_10/attention/output/dense/bias3bert/encoder/layer_10/attention/output/dense/kernel-bert/encoder/layer_10/attention/self/key/bias/bert/encoder/layer_10/attention/self/key/kernel/bert/encoder/layer_10/attention/self/query/bias1bert/encoder/layer_10/attention/self/query/kernel/bert/encoder/layer_10/attention/self/value/bias1bert/encoder/layer_10/attention/self/value/kernel-bert/encoder/layer_10/intermediate/dense/bias/bert/encoder/layer_10/intermediate/dense/kernel+bert/encoder/layer_10/output/LayerNorm/beta,bert/encoder/layer_10/output/LayerNorm/gamma'bert/encoder/layer_10/output/dense/bias)bert/encoder/layer_10/output/dense/kernel5bert/encoder/layer_11/attention/output/LayerNorm/beta6bert/encoder/layer_11/attention/output/LayerNorm/gamma1bert/encoder/layer_11/attention/output/dense/bias3bert/encoder/layer_11/attention/output/dense/kernel-bert/encoder/layer_11/attention/self/key/bias/bert/encoder/layer_11/attention/self/key/kernel/bert/encoder/layer_11/attention/self/query/bias1bert/encoder/layer_11/attention/self/query/kernel/bert/encoder/layer_11/attention/self/value/bias1bert/encoder/layer_11/attention/self/value/kernel-bert/encoder/layer_11/intermediate/dense/bias/bert/encoder/layer_11/intermediate/dense/kernel+bert/encoder/layer_11/output/LayerNorm/beta,bert/encoder/layer_11/output/LayerNorm/gamma'bert/encoder/layer_11/output/dense/bias)bert/encoder/layer_11/output/dense/kernel4bert/encoder/layer_2/attention/output/LayerNorm/beta5bert/encoder/layer_2/attention/output/LayerNorm/gamma0bert/encoder/layer_2/attention/output/dense/bias2bert/encoder/layer_2/attention/output/dense/kernel,bert/encoder/layer_2/attention/self/key/bias.bert/encoder/layer_2/attention/self/key/kernel.bert/encoder/layer_2/attention/self/query/bias0bert/encoder/layer_2/attention/self/query/kernel.bert/encoder/layer_2/attention/self/value/bias0bert/encoder/layer_2/attention/self/value/kernel,bert/encoder/layer_2/intermediate/dense/bias.bert/encoder/layer_2/intermediate/dense/kernel*bert/encoder/layer_2/output/LayerNorm/beta+bert/encoder/layer_2/output/LayerNorm/gamma&bert/encoder/layer_2/output/dense/bias(bert/encoder/layer_2/output/dense/kernel4bert/encoder/layer_3/attention/output/LayerNorm/beta5bert/encoder/layer_3/attention/output/LayerNorm/gamma0bert/encoder/layer_3/attention/output/dense/bias2bert/encoder/layer_3/attention/output/dense/kernel,bert/encoder/layer_3/attention/self/key/bias.bert/encoder/layer_3/attention/self/key/kernel.bert/encoder/layer_3/attention/self/query/bias0bert/encoder/layer_3/attention/self/query/kernel.bert/encoder/layer_3/attention/self/value/bias0bert/encoder/layer_3/attention/self/value/kernel,bert/encoder/layer_3/intermediate/dense/bias.bert/encoder/layer_3/intermediate/dense/kernel*bert/encoder/layer_3/output/LayerNorm/beta+bert/encoder/layer_3/output/LayerNorm/gamma&bert/encoder/layer_3/output/dense/bias(bert/encoder/layer_3/output/dense/kernel4bert/encoder/layer_4/attention/output/LayerNorm/beta5bert/encoder/layer_4/attention/output/LayerNorm/gamma0bert/encoder/layer_4/attention/output/dense/bias2bert/encoder/layer_4/attention/output/dense/kernel,bert/encoder/layer_4/attention/self/key/bias.bert/encoder/layer_4/attention/self/key/kernel.bert/encoder/layer_4/attention/self/query/bias0bert/encoder/layer_4/attention/self/query/kernel.bert/encoder/layer_4/attention/self/value/bias0bert/encoder/layer_4/attention/self/value/kernel,bert/encoder/layer_4/intermediate/dense/bias.bert/encoder/layer_4/intermediate/dense/kernel*bert/encoder/layer_4/output/LayerNorm/beta+bert/encoder/layer_4/output/LayerNorm/gamma&bert/encoder/layer_4/output/dense/bias(bert/encoder/layer_4/output/dense/kernel4bert/encoder/layer_5/attention/output/LayerNorm/beta5bert/encoder/layer_5/attention/output/LayerNorm/gamma0bert/encoder/layer_5/attention/output/dense/bias2bert/encoder/layer_5/attention/output/dense/kernel,bert/encoder/layer_5/attention/self/key/bias.bert/encoder/layer_5/attention/self/key/kernel.bert/encoder/layer_5/attention/self/query/bias0bert/encoder/layer_5/attention/self/query/kernel.bert/encoder/layer_5/attention/self/value/bias0bert/encoder/layer_5/attention/self/value/kernel,bert/encoder/layer_5/intermediate/dense/bias.bert/encoder/layer_5/intermediate/dense/kernel*bert/encoder/layer_5/output/LayerNorm/beta+bert/encoder/layer_5/output/LayerNorm/gamma&bert/encoder/layer_5/output/dense/bias(bert/encoder/layer_5/output/dense/kernel4bert/encoder/layer_6/attention/output/LayerNorm/beta5bert/encoder/layer_6/attention/output/LayerNorm/gamma0bert/encoder/layer_6/attention/output/dense/bias2bert/encoder/layer_6/attention/output/dense/kernel,bert/encoder/layer_6/attention/self/key/bias.bert/encoder/layer_6/attention/self/key/kernel.bert/encoder/layer_6/attention/self/query/bias0bert/encoder/layer_6/attention/self/query/kernel.bert/encoder/layer_6/attention/self/value/bias0bert/encoder/layer_6/attention/self/value/kernel,bert/encoder/layer_6/intermediate/dense/bias.bert/encoder/layer_6/intermediate/dense/kernel*bert/encoder/layer_6/output/LayerNorm/beta+bert/encoder/layer_6/output/LayerNorm/gamma&bert/encoder/layer_6/output/dense/bias(bert/encoder/layer_6/output/dense/kernel4bert/encoder/layer_7/attention/output/LayerNorm/beta5bert/encoder/layer_7/attention/output/LayerNorm/gamma0bert/encoder/layer_7/attention/output/dense/bias2bert/encoder/layer_7/attention/output/dense/kernel,bert/encoder/layer_7/attention/self/key/bias.bert/encoder/layer_7/attention/self/key/kernel.bert/encoder/layer_7/attention/self/query/bias0bert/encoder/layer_7/attention/self/query/kernel.bert/encoder/layer_7/attention/self/value/bias0bert/encoder/layer_7/attention/self/value/kernel,bert/encoder/layer_7/intermediate/dense/bias.bert/encoder/layer_7/intermediate/dense/kernel*bert/encoder/layer_7/output/LayerNorm/beta+bert/encoder/layer_7/output/LayerNorm/gamma&bert/encoder/layer_7/output/dense/bias(bert/encoder/layer_7/output/dense/kernel4bert/encoder/layer_8/attention/output/LayerNorm/beta5bert/encoder/layer_8/attention/output/LayerNorm/gamma0bert/encoder/layer_8/attention/output/dense/bias2bert/encoder/layer_8/attention/output/dense/kernel,bert/encoder/layer_8/attention/self/key/bias.bert/encoder/layer_8/attention/self/key/kernel.bert/encoder/layer_8/attention/self/query/bias0bert/encoder/layer_8/attention/self/query/kernel.bert/encoder/layer_8/attention/self/value/bias0bert/encoder/layer_8/attention/self/value/kernel,bert/encoder/layer_8/intermediate/dense/bias.bert/encoder/layer_8/intermediate/dense/kernel*bert/encoder/layer_8/output/LayerNorm/beta+bert/encoder/layer_8/output/LayerNorm/gamma&bert/encoder/layer_8/output/dense/bias(bert/encoder/layer_8/output/dense/kernel4bert/encoder/layer_9/attention/output/LayerNorm/beta5bert/encoder/layer_9/attention/output/LayerNorm/gamma0bert/encoder/layer_9/attention/output/dense/bias2bert/encoder/layer_9/attention/output/dense/kernel,bert/encoder/layer_9/attention/self/key/bias.bert/encoder/layer_9/attention/self/key/kernel.bert/encoder/layer_9/attention/self/query/bias0bert/encoder/layer_9/attention/self/query/kernel.bert/encoder/layer_9/attention/self/value/bias0bert/encoder/layer_9/attention/self/value/kernel,bert/encoder/layer_9/intermediate/dense/bias.bert/encoder/layer_9/intermediate/dense/kernel*bert/encoder/layer_9/output/LayerNorm/beta+bert/encoder/layer_9/output/LayerNorm/gamma&bert/encoder/layer_9/output/dense/bias(bert/encoder/layer_9/output/dense/kernelbert/pooler/dense/biasbert/pooler/dense/kerneloutput_biasoutput_weights*?
dtypes?
?2?
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
_output_shapes
: *
T0
?J
save_1/RestoreV2/tensor_namesConst*?J
value?JB?J?Bbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelB5bert/encoder/layer_10/attention/output/LayerNorm/betaB6bert/encoder/layer_10/attention/output/LayerNorm/gammaB1bert/encoder/layer_10/attention/output/dense/biasB3bert/encoder/layer_10/attention/output/dense/kernelB-bert/encoder/layer_10/attention/self/key/biasB/bert/encoder/layer_10/attention/self/key/kernelB/bert/encoder/layer_10/attention/self/query/biasB1bert/encoder/layer_10/attention/self/query/kernelB/bert/encoder/layer_10/attention/self/value/biasB1bert/encoder/layer_10/attention/self/value/kernelB-bert/encoder/layer_10/intermediate/dense/biasB/bert/encoder/layer_10/intermediate/dense/kernelB+bert/encoder/layer_10/output/LayerNorm/betaB,bert/encoder/layer_10/output/LayerNorm/gammaB'bert/encoder/layer_10/output/dense/biasB)bert/encoder/layer_10/output/dense/kernelB5bert/encoder/layer_11/attention/output/LayerNorm/betaB6bert/encoder/layer_11/attention/output/LayerNorm/gammaB1bert/encoder/layer_11/attention/output/dense/biasB3bert/encoder/layer_11/attention/output/dense/kernelB-bert/encoder/layer_11/attention/self/key/biasB/bert/encoder/layer_11/attention/self/key/kernelB/bert/encoder/layer_11/attention/self/query/biasB1bert/encoder/layer_11/attention/self/query/kernelB/bert/encoder/layer_11/attention/self/value/biasB1bert/encoder/layer_11/attention/self/value/kernelB-bert/encoder/layer_11/intermediate/dense/biasB/bert/encoder/layer_11/intermediate/dense/kernelB+bert/encoder/layer_11/output/LayerNorm/betaB,bert/encoder/layer_11/output/LayerNorm/gammaB'bert/encoder/layer_11/output/dense/biasB)bert/encoder/layer_11/output/dense/kernelB4bert/encoder/layer_2/attention/output/LayerNorm/betaB5bert/encoder/layer_2/attention/output/LayerNorm/gammaB0bert/encoder/layer_2/attention/output/dense/biasB2bert/encoder/layer_2/attention/output/dense/kernelB,bert/encoder/layer_2/attention/self/key/biasB.bert/encoder/layer_2/attention/self/key/kernelB.bert/encoder/layer_2/attention/self/query/biasB0bert/encoder/layer_2/attention/self/query/kernelB.bert/encoder/layer_2/attention/self/value/biasB0bert/encoder/layer_2/attention/self/value/kernelB,bert/encoder/layer_2/intermediate/dense/biasB.bert/encoder/layer_2/intermediate/dense/kernelB*bert/encoder/layer_2/output/LayerNorm/betaB+bert/encoder/layer_2/output/LayerNorm/gammaB&bert/encoder/layer_2/output/dense/biasB(bert/encoder/layer_2/output/dense/kernelB4bert/encoder/layer_3/attention/output/LayerNorm/betaB5bert/encoder/layer_3/attention/output/LayerNorm/gammaB0bert/encoder/layer_3/attention/output/dense/biasB2bert/encoder/layer_3/attention/output/dense/kernelB,bert/encoder/layer_3/attention/self/key/biasB.bert/encoder/layer_3/attention/self/key/kernelB.bert/encoder/layer_3/attention/self/query/biasB0bert/encoder/layer_3/attention/self/query/kernelB.bert/encoder/layer_3/attention/self/value/biasB0bert/encoder/layer_3/attention/self/value/kernelB,bert/encoder/layer_3/intermediate/dense/biasB.bert/encoder/layer_3/intermediate/dense/kernelB*bert/encoder/layer_3/output/LayerNorm/betaB+bert/encoder/layer_3/output/LayerNorm/gammaB&bert/encoder/layer_3/output/dense/biasB(bert/encoder/layer_3/output/dense/kernelB4bert/encoder/layer_4/attention/output/LayerNorm/betaB5bert/encoder/layer_4/attention/output/LayerNorm/gammaB0bert/encoder/layer_4/attention/output/dense/biasB2bert/encoder/layer_4/attention/output/dense/kernelB,bert/encoder/layer_4/attention/self/key/biasB.bert/encoder/layer_4/attention/self/key/kernelB.bert/encoder/layer_4/attention/self/query/biasB0bert/encoder/layer_4/attention/self/query/kernelB.bert/encoder/layer_4/attention/self/value/biasB0bert/encoder/layer_4/attention/self/value/kernelB,bert/encoder/layer_4/intermediate/dense/biasB.bert/encoder/layer_4/intermediate/dense/kernelB*bert/encoder/layer_4/output/LayerNorm/betaB+bert/encoder/layer_4/output/LayerNorm/gammaB&bert/encoder/layer_4/output/dense/biasB(bert/encoder/layer_4/output/dense/kernelB4bert/encoder/layer_5/attention/output/LayerNorm/betaB5bert/encoder/layer_5/attention/output/LayerNorm/gammaB0bert/encoder/layer_5/attention/output/dense/biasB2bert/encoder/layer_5/attention/output/dense/kernelB,bert/encoder/layer_5/attention/self/key/biasB.bert/encoder/layer_5/attention/self/key/kernelB.bert/encoder/layer_5/attention/self/query/biasB0bert/encoder/layer_5/attention/self/query/kernelB.bert/encoder/layer_5/attention/self/value/biasB0bert/encoder/layer_5/attention/self/value/kernelB,bert/encoder/layer_5/intermediate/dense/biasB.bert/encoder/layer_5/intermediate/dense/kernelB*bert/encoder/layer_5/output/LayerNorm/betaB+bert/encoder/layer_5/output/LayerNorm/gammaB&bert/encoder/layer_5/output/dense/biasB(bert/encoder/layer_5/output/dense/kernelB4bert/encoder/layer_6/attention/output/LayerNorm/betaB5bert/encoder/layer_6/attention/output/LayerNorm/gammaB0bert/encoder/layer_6/attention/output/dense/biasB2bert/encoder/layer_6/attention/output/dense/kernelB,bert/encoder/layer_6/attention/self/key/biasB.bert/encoder/layer_6/attention/self/key/kernelB.bert/encoder/layer_6/attention/self/query/biasB0bert/encoder/layer_6/attention/self/query/kernelB.bert/encoder/layer_6/attention/self/value/biasB0bert/encoder/layer_6/attention/self/value/kernelB,bert/encoder/layer_6/intermediate/dense/biasB.bert/encoder/layer_6/intermediate/dense/kernelB*bert/encoder/layer_6/output/LayerNorm/betaB+bert/encoder/layer_6/output/LayerNorm/gammaB&bert/encoder/layer_6/output/dense/biasB(bert/encoder/layer_6/output/dense/kernelB4bert/encoder/layer_7/attention/output/LayerNorm/betaB5bert/encoder/layer_7/attention/output/LayerNorm/gammaB0bert/encoder/layer_7/attention/output/dense/biasB2bert/encoder/layer_7/attention/output/dense/kernelB,bert/encoder/layer_7/attention/self/key/biasB.bert/encoder/layer_7/attention/self/key/kernelB.bert/encoder/layer_7/attention/self/query/biasB0bert/encoder/layer_7/attention/self/query/kernelB.bert/encoder/layer_7/attention/self/value/biasB0bert/encoder/layer_7/attention/self/value/kernelB,bert/encoder/layer_7/intermediate/dense/biasB.bert/encoder/layer_7/intermediate/dense/kernelB*bert/encoder/layer_7/output/LayerNorm/betaB+bert/encoder/layer_7/output/LayerNorm/gammaB&bert/encoder/layer_7/output/dense/biasB(bert/encoder/layer_7/output/dense/kernelB4bert/encoder/layer_8/attention/output/LayerNorm/betaB5bert/encoder/layer_8/attention/output/LayerNorm/gammaB0bert/encoder/layer_8/attention/output/dense/biasB2bert/encoder/layer_8/attention/output/dense/kernelB,bert/encoder/layer_8/attention/self/key/biasB.bert/encoder/layer_8/attention/self/key/kernelB.bert/encoder/layer_8/attention/self/query/biasB0bert/encoder/layer_8/attention/self/query/kernelB.bert/encoder/layer_8/attention/self/value/biasB0bert/encoder/layer_8/attention/self/value/kernelB,bert/encoder/layer_8/intermediate/dense/biasB.bert/encoder/layer_8/intermediate/dense/kernelB*bert/encoder/layer_8/output/LayerNorm/betaB+bert/encoder/layer_8/output/LayerNorm/gammaB&bert/encoder/layer_8/output/dense/biasB(bert/encoder/layer_8/output/dense/kernelB4bert/encoder/layer_9/attention/output/LayerNorm/betaB5bert/encoder/layer_9/attention/output/LayerNorm/gammaB0bert/encoder/layer_9/attention/output/dense/biasB2bert/encoder/layer_9/attention/output/dense/kernelB,bert/encoder/layer_9/attention/self/key/biasB.bert/encoder/layer_9/attention/self/key/kernelB.bert/encoder/layer_9/attention/self/query/biasB0bert/encoder/layer_9/attention/self/query/kernelB.bert/encoder/layer_9/attention/self/value/biasB0bert/encoder/layer_9/attention/self/value/kernelB,bert/encoder/layer_9/intermediate/dense/biasB.bert/encoder/layer_9/intermediate/dense/kernelB*bert/encoder/layer_9/output/LayerNorm/betaB+bert/encoder/layer_9/output/LayerNorm/gammaB&bert/encoder/layer_9/output/dense/biasB(bert/encoder/layer_9/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBoutput_biasBoutput_weights*
dtype0*
_output_shapes	
:?
?
!save_1/RestoreV2/shape_and_slicesConst*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:?
?	
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?
?
save_1/AssignAssignbert/embeddings/LayerNorm/betasave_1/RestoreV2*
T0*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_1Assignbert/embeddings/LayerNorm/gammasave_1/RestoreV2:1*
use_locking(*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_2Assign#bert/embeddings/position_embeddingssave_1/RestoreV2:2* 
_output_shapes
:
??*
use_locking(*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings*
validate_shape(
?
save_1/Assign_3Assign%bert/embeddings/token_type_embeddingssave_1/RestoreV2:3*
_output_shapes
:	?*
use_locking(*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
validate_shape(
?
save_1/Assign_4Assignbert/embeddings/word_embeddingssave_1/RestoreV2:4*
use_locking(*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*
validate_shape(*!
_output_shapes
:???
?
save_1/Assign_5Assign4bert/encoder/layer_0/attention/output/LayerNorm/betasave_1/RestoreV2:5*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_6Assign5bert/encoder/layer_0/attention/output/LayerNorm/gammasave_1/RestoreV2:6*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_7Assign0bert/encoder/layer_0/attention/output/dense/biassave_1/RestoreV2:7*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_8Assign2bert/encoder/layer_0/attention/output/dense/kernelsave_1/RestoreV2:8*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_9Assign,bert/encoder/layer_0/attention/self/key/biassave_1/RestoreV2:9*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_10Assign.bert/encoder/layer_0/attention/self/key/kernelsave_1/RestoreV2:10*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_11Assign.bert/encoder/layer_0/attention/self/query/biassave_1/RestoreV2:11*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_12Assign0bert/encoder/layer_0/attention/self/query/kernelsave_1/RestoreV2:12*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_13Assign.bert/encoder/layer_0/attention/self/value/biassave_1/RestoreV2:13*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
validate_shape(
?
save_1/Assign_14Assign0bert/encoder/layer_0/attention/self/value/kernelsave_1/RestoreV2:14*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel
?
save_1/Assign_15Assign,bert/encoder/layer_0/intermediate/dense/biassave_1/RestoreV2:15*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_16Assign.bert/encoder/layer_0/intermediate/dense/kernelsave_1/RestoreV2:16*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_17Assign*bert/encoder/layer_0/output/LayerNorm/betasave_1/RestoreV2:17*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta
?
save_1/Assign_18Assign+bert/encoder/layer_0/output/LayerNorm/gammasave_1/RestoreV2:18*
T0*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_19Assign&bert/encoder/layer_0/output/dense/biassave_1/RestoreV2:19*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias
?
save_1/Assign_20Assign(bert/encoder/layer_0/output/dense/kernelsave_1/RestoreV2:20*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_21Assign4bert/encoder/layer_1/attention/output/LayerNorm/betasave_1/RestoreV2:21*
T0*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_22Assign5bert/encoder/layer_1/attention/output/LayerNorm/gammasave_1/RestoreV2:22*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_23Assign0bert/encoder/layer_1/attention/output/dense/biassave_1/RestoreV2:23*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
validate_shape(
?
save_1/Assign_24Assign2bert/encoder/layer_1/attention/output/dense/kernelsave_1/RestoreV2:24*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel
?
save_1/Assign_25Assign,bert/encoder/layer_1/attention/self/key/biassave_1/RestoreV2:25*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
validate_shape(
?
save_1/Assign_26Assign.bert/encoder/layer_1/attention/self/key/kernelsave_1/RestoreV2:26*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_27Assign.bert/encoder/layer_1/attention/self/query/biassave_1/RestoreV2:27*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias
?
save_1/Assign_28Assign0bert/encoder/layer_1/attention/self/query/kernelsave_1/RestoreV2:28*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_29Assign.bert/encoder/layer_1/attention/self/value/biassave_1/RestoreV2:29*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias
?
save_1/Assign_30Assign0bert/encoder/layer_1/attention/self/value/kernelsave_1/RestoreV2:30*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_31Assign,bert/encoder/layer_1/intermediate/dense/biassave_1/RestoreV2:31*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_32Assign.bert/encoder/layer_1/intermediate/dense/kernelsave_1/RestoreV2:32*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_33Assign*bert/encoder/layer_1/output/LayerNorm/betasave_1/RestoreV2:33*
T0*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_34Assign+bert/encoder/layer_1/output/LayerNorm/gammasave_1/RestoreV2:34*
T0*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_35Assign&bert/encoder/layer_1/output/dense/biassave_1/RestoreV2:35*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_36Assign(bert/encoder/layer_1/output/dense/kernelsave_1/RestoreV2:36*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel
?
save_1/Assign_37Assign5bert/encoder/layer_10/attention/output/LayerNorm/betasave_1/RestoreV2:37*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_38Assign6bert/encoder/layer_10/attention/output/LayerNorm/gammasave_1/RestoreV2:38*
_output_shapes	
:?*
use_locking(*
T0*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
validate_shape(
?
save_1/Assign_39Assign1bert/encoder/layer_10/attention/output/dense/biassave_1/RestoreV2:39*
_output_shapes	
:?*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias*
validate_shape(
?
save_1/Assign_40Assign3bert/encoder/layer_10/attention/output/dense/kernelsave_1/RestoreV2:40*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_41Assign-bert/encoder/layer_10/attention/self/key/biassave_1/RestoreV2:41*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_42Assign/bert/encoder/layer_10/attention/self/key/kernelsave_1/RestoreV2:42*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_43Assign/bert/encoder/layer_10/attention/self/query/biassave_1/RestoreV2:43*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_44Assign1bert/encoder/layer_10/attention/self/query/kernelsave_1/RestoreV2:44*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_45Assign/bert/encoder/layer_10/attention/self/value/biassave_1/RestoreV2:45*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_46Assign1bert/encoder/layer_10/attention/self/value/kernelsave_1/RestoreV2:46*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_47Assign-bert/encoder/layer_10/intermediate/dense/biassave_1/RestoreV2:47*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_48Assign/bert/encoder/layer_10/intermediate/dense/kernelsave_1/RestoreV2:48*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_49Assign+bert/encoder/layer_10/output/LayerNorm/betasave_1/RestoreV2:49*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_50Assign,bert/encoder/layer_10/output/LayerNorm/gammasave_1/RestoreV2:50*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_51Assign'bert/encoder/layer_10/output/dense/biassave_1/RestoreV2:51*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_52Assign)bert/encoder/layer_10/output/dense/kernelsave_1/RestoreV2:52*
use_locking(*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_53Assign5bert/encoder/layer_11/attention/output/LayerNorm/betasave_1/RestoreV2:53*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_54Assign6bert/encoder/layer_11/attention/output/LayerNorm/gammasave_1/RestoreV2:54*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma
?
save_1/Assign_55Assign1bert/encoder/layer_11/attention/output/dense/biassave_1/RestoreV2:55*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_56Assign3bert/encoder/layer_11/attention/output/dense/kernelsave_1/RestoreV2:56*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_57Assign-bert/encoder/layer_11/attention/self/key/biassave_1/RestoreV2:57*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias
?
save_1/Assign_58Assign/bert/encoder/layer_11/attention/self/key/kernelsave_1/RestoreV2:58* 
_output_shapes
:
??*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
validate_shape(
?
save_1/Assign_59Assign/bert/encoder/layer_11/attention/self/query/biassave_1/RestoreV2:59*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_60Assign1bert/encoder/layer_11/attention/self/query/kernelsave_1/RestoreV2:60*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel
?
save_1/Assign_61Assign/bert/encoder/layer_11/attention/self/value/biassave_1/RestoreV2:61*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_62Assign1bert/encoder/layer_11/attention/self/value/kernelsave_1/RestoreV2:62*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_63Assign-bert/encoder/layer_11/intermediate/dense/biassave_1/RestoreV2:63*
T0*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_64Assign/bert/encoder/layer_11/intermediate/dense/kernelsave_1/RestoreV2:64*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_65Assign+bert/encoder/layer_11/output/LayerNorm/betasave_1/RestoreV2:65*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_66Assign,bert/encoder/layer_11/output/LayerNorm/gammasave_1/RestoreV2:66*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
validate_shape(
?
save_1/Assign_67Assign'bert/encoder/layer_11/output/dense/biassave_1/RestoreV2:67*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_68Assign)bert/encoder/layer_11/output/dense/kernelsave_1/RestoreV2:68*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_69Assign4bert/encoder/layer_2/attention/output/LayerNorm/betasave_1/RestoreV2:69*
_output_shapes	
:?*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
validate_shape(
?
save_1/Assign_70Assign5bert/encoder/layer_2/attention/output/LayerNorm/gammasave_1/RestoreV2:70*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_71Assign0bert/encoder/layer_2/attention/output/dense/biassave_1/RestoreV2:71*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_72Assign2bert/encoder/layer_2/attention/output/dense/kernelsave_1/RestoreV2:72*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_73Assign,bert/encoder/layer_2/attention/self/key/biassave_1/RestoreV2:73*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_74Assign.bert/encoder/layer_2/attention/self/key/kernelsave_1/RestoreV2:74*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_75Assign.bert/encoder/layer_2/attention/self/query/biassave_1/RestoreV2:75*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_76Assign0bert/encoder/layer_2/attention/self/query/kernelsave_1/RestoreV2:76*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_77Assign.bert/encoder/layer_2/attention/self/value/biassave_1/RestoreV2:77*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_78Assign0bert/encoder/layer_2/attention/self/value/kernelsave_1/RestoreV2:78*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel
?
save_1/Assign_79Assign,bert/encoder/layer_2/intermediate/dense/biassave_1/RestoreV2:79*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_80Assign.bert/encoder/layer_2/intermediate/dense/kernelsave_1/RestoreV2:80*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_81Assign*bert/encoder/layer_2/output/LayerNorm/betasave_1/RestoreV2:81*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_82Assign+bert/encoder/layer_2/output/LayerNorm/gammasave_1/RestoreV2:82*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_83Assign&bert/encoder/layer_2/output/dense/biassave_1/RestoreV2:83*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_84Assign(bert/encoder/layer_2/output/dense/kernelsave_1/RestoreV2:84*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_85Assign4bert/encoder/layer_3/attention/output/LayerNorm/betasave_1/RestoreV2:85*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_86Assign5bert/encoder/layer_3/attention/output/LayerNorm/gammasave_1/RestoreV2:86*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
validate_shape(
?
save_1/Assign_87Assign0bert/encoder/layer_3/attention/output/dense/biassave_1/RestoreV2:87*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_88Assign2bert/encoder/layer_3/attention/output/dense/kernelsave_1/RestoreV2:88* 
_output_shapes
:
??*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
validate_shape(
?
save_1/Assign_89Assign,bert/encoder/layer_3/attention/self/key/biassave_1/RestoreV2:89*
T0*?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_90Assign.bert/encoder/layer_3/attention/self/key/kernelsave_1/RestoreV2:90*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_91Assign.bert/encoder/layer_3/attention/self/query/biassave_1/RestoreV2:91*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_92Assign0bert/encoder/layer_3/attention/self/query/kernelsave_1/RestoreV2:92*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_93Assign.bert/encoder/layer_3/attention/self/value/biassave_1/RestoreV2:93*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_94Assign0bert/encoder/layer_3/attention/self/value/kernelsave_1/RestoreV2:94*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_95Assign,bert/encoder/layer_3/intermediate/dense/biassave_1/RestoreV2:95*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_96Assign.bert/encoder/layer_3/intermediate/dense/kernelsave_1/RestoreV2:96*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel
?
save_1/Assign_97Assign*bert/encoder/layer_3/output/LayerNorm/betasave_1/RestoreV2:97*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta
?
save_1/Assign_98Assign+bert/encoder/layer_3/output/LayerNorm/gammasave_1/RestoreV2:98*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_99Assign&bert/encoder/layer_3/output/dense/biassave_1/RestoreV2:99*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_100Assign(bert/encoder/layer_3/output/dense/kernelsave_1/RestoreV2:100*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save_1/Assign_101Assign4bert/encoder/layer_4/attention/output/LayerNorm/betasave_1/RestoreV2:101*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_102Assign5bert/encoder/layer_4/attention/output/LayerNorm/gammasave_1/RestoreV2:102*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_103Assign0bert/encoder/layer_4/attention/output/dense/biassave_1/RestoreV2:103*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_104Assign2bert/encoder/layer_4/attention/output/dense/kernelsave_1/RestoreV2:104*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_105Assign,bert/encoder/layer_4/attention/self/key/biassave_1/RestoreV2:105*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_106Assign.bert/encoder/layer_4/attention/self/key/kernelsave_1/RestoreV2:106*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_107Assign.bert/encoder/layer_4/attention/self/query/biassave_1/RestoreV2:107*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_108Assign0bert/encoder/layer_4/attention/self/query/kernelsave_1/RestoreV2:108*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel
?
save_1/Assign_109Assign.bert/encoder/layer_4/attention/self/value/biassave_1/RestoreV2:109*A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_110Assign0bert/encoder/layer_4/attention/self/value/kernelsave_1/RestoreV2:110*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel
?
save_1/Assign_111Assign,bert/encoder/layer_4/intermediate/dense/biassave_1/RestoreV2:111*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_112Assign.bert/encoder/layer_4/intermediate/dense/kernelsave_1/RestoreV2:112*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_113Assign*bert/encoder/layer_4/output/LayerNorm/betasave_1/RestoreV2:113*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
validate_shape(
?
save_1/Assign_114Assign+bert/encoder/layer_4/output/LayerNorm/gammasave_1/RestoreV2:114*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_115Assign&bert/encoder/layer_4/output/dense/biassave_1/RestoreV2:115*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
validate_shape(
?
save_1/Assign_116Assign(bert/encoder/layer_4/output/dense/kernelsave_1/RestoreV2:116*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_117Assign4bert/encoder/layer_5/attention/output/LayerNorm/betasave_1/RestoreV2:117*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_118Assign5bert/encoder/layer_5/attention/output/LayerNorm/gammasave_1/RestoreV2:118*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_119Assign0bert/encoder/layer_5/attention/output/dense/biassave_1/RestoreV2:119*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_120Assign2bert/encoder/layer_5/attention/output/dense/kernelsave_1/RestoreV2:120*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_121Assign,bert/encoder/layer_5/attention/self/key/biassave_1/RestoreV2:121*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_122Assign.bert/encoder/layer_5/attention/self/key/kernelsave_1/RestoreV2:122*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_123Assign.bert/encoder/layer_5/attention/self/query/biassave_1/RestoreV2:123*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_124Assign0bert/encoder/layer_5/attention/self/query/kernelsave_1/RestoreV2:124*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel
?
save_1/Assign_125Assign.bert/encoder/layer_5/attention/self/value/biassave_1/RestoreV2:125*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_126Assign0bert/encoder/layer_5/attention/self/value/kernelsave_1/RestoreV2:126*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel
?
save_1/Assign_127Assign,bert/encoder/layer_5/intermediate/dense/biassave_1/RestoreV2:127*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias
?
save_1/Assign_128Assign.bert/encoder/layer_5/intermediate/dense/kernelsave_1/RestoreV2:128*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_129Assign*bert/encoder/layer_5/output/LayerNorm/betasave_1/RestoreV2:129*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_130Assign+bert/encoder/layer_5/output/LayerNorm/gammasave_1/RestoreV2:130*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_131Assign&bert/encoder/layer_5/output/dense/biassave_1/RestoreV2:131*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_132Assign(bert/encoder/layer_5/output/dense/kernelsave_1/RestoreV2:132* 
_output_shapes
:
??*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
validate_shape(
?
save_1/Assign_133Assign4bert/encoder/layer_6/attention/output/LayerNorm/betasave_1/RestoreV2:133*G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_134Assign5bert/encoder/layer_6/attention/output/LayerNorm/gammasave_1/RestoreV2:134*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma
?
save_1/Assign_135Assign0bert/encoder/layer_6/attention/output/dense/biassave_1/RestoreV2:135*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
validate_shape(
?
save_1/Assign_136Assign2bert/encoder/layer_6/attention/output/dense/kernelsave_1/RestoreV2:136*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_137Assign,bert/encoder/layer_6/attention/self/key/biassave_1/RestoreV2:137*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias
?
save_1/Assign_138Assign.bert/encoder/layer_6/attention/self/key/kernelsave_1/RestoreV2:138* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
validate_shape(
?
save_1/Assign_139Assign.bert/encoder/layer_6/attention/self/query/biassave_1/RestoreV2:139*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_140Assign0bert/encoder/layer_6/attention/self/query/kernelsave_1/RestoreV2:140*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_141Assign.bert/encoder/layer_6/attention/self/value/biassave_1/RestoreV2:141*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias
?
save_1/Assign_142Assign0bert/encoder/layer_6/attention/self/value/kernelsave_1/RestoreV2:142*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_143Assign,bert/encoder/layer_6/intermediate/dense/biassave_1/RestoreV2:143*
T0*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_144Assign.bert/encoder/layer_6/intermediate/dense/kernelsave_1/RestoreV2:144*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_145Assign*bert/encoder/layer_6/output/LayerNorm/betasave_1/RestoreV2:145*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta
?
save_1/Assign_146Assign+bert/encoder/layer_6/output/LayerNorm/gammasave_1/RestoreV2:146*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_147Assign&bert/encoder/layer_6/output/dense/biassave_1/RestoreV2:147*9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_148Assign(bert/encoder/layer_6/output/dense/kernelsave_1/RestoreV2:148*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel
?
save_1/Assign_149Assign4bert/encoder/layer_7/attention/output/LayerNorm/betasave_1/RestoreV2:149*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_150Assign5bert/encoder/layer_7/attention/output/LayerNorm/gammasave_1/RestoreV2:150*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
validate_shape(
?
save_1/Assign_151Assign0bert/encoder/layer_7/attention/output/dense/biassave_1/RestoreV2:151*C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_152Assign2bert/encoder/layer_7/attention/output/dense/kernelsave_1/RestoreV2:152*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_153Assign,bert/encoder/layer_7/attention/self/key/biassave_1/RestoreV2:153*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_154Assign.bert/encoder/layer_7/attention/self/key/kernelsave_1/RestoreV2:154*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_155Assign.bert/encoder/layer_7/attention/self/query/biassave_1/RestoreV2:155*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_156Assign0bert/encoder/layer_7/attention/self/query/kernelsave_1/RestoreV2:156*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_157Assign.bert/encoder/layer_7/attention/self/value/biassave_1/RestoreV2:157*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_158Assign0bert/encoder/layer_7/attention/self/value/kernelsave_1/RestoreV2:158*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_159Assign,bert/encoder/layer_7/intermediate/dense/biassave_1/RestoreV2:159*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_160Assign.bert/encoder/layer_7/intermediate/dense/kernelsave_1/RestoreV2:160*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(
?
save_1/Assign_161Assign*bert/encoder/layer_7/output/LayerNorm/betasave_1/RestoreV2:161*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
validate_shape(
?
save_1/Assign_162Assign+bert/encoder/layer_7/output/LayerNorm/gammasave_1/RestoreV2:162*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_163Assign&bert/encoder/layer_7/output/dense/biassave_1/RestoreV2:163*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_164Assign(bert/encoder/layer_7/output/dense/kernelsave_1/RestoreV2:164*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_165Assign4bert/encoder/layer_8/attention/output/LayerNorm/betasave_1/RestoreV2:165*
T0*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_166Assign5bert/encoder/layer_8/attention/output/LayerNorm/gammasave_1/RestoreV2:166*
_output_shapes	
:?*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
validate_shape(
?
save_1/Assign_167Assign0bert/encoder/layer_8/attention/output/dense/biassave_1/RestoreV2:167*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_168Assign2bert/encoder/layer_8/attention/output/dense/kernelsave_1/RestoreV2:168*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_169Assign,bert/encoder/layer_8/attention/self/key/biassave_1/RestoreV2:169*?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_170Assign.bert/encoder/layer_8/attention/self/key/kernelsave_1/RestoreV2:170*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save_1/Assign_171Assign.bert/encoder/layer_8/attention/self/query/biassave_1/RestoreV2:171*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias
?
save_1/Assign_172Assign0bert/encoder/layer_8/attention/self/query/kernelsave_1/RestoreV2:172*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel
?
save_1/Assign_173Assign.bert/encoder/layer_8/attention/self/value/biassave_1/RestoreV2:173*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias
?
save_1/Assign_174Assign0bert/encoder/layer_8/attention/self/value/kernelsave_1/RestoreV2:174*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_175Assign,bert/encoder/layer_8/intermediate/dense/biassave_1/RestoreV2:175*
_output_shapes	
:?*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
validate_shape(
?
save_1/Assign_176Assign.bert/encoder/layer_8/intermediate/dense/kernelsave_1/RestoreV2:176*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel
?
save_1/Assign_177Assign*bert/encoder/layer_8/output/LayerNorm/betasave_1/RestoreV2:177*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_178Assign+bert/encoder/layer_8/output/LayerNorm/gammasave_1/RestoreV2:178*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_179Assign&bert/encoder/layer_8/output/dense/biassave_1/RestoreV2:179*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias
?
save_1/Assign_180Assign(bert/encoder/layer_8/output/dense/kernelsave_1/RestoreV2:180*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_181Assign4bert/encoder/layer_9/attention/output/LayerNorm/betasave_1/RestoreV2:181*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_182Assign5bert/encoder/layer_9/attention/output/LayerNorm/gammasave_1/RestoreV2:182*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_183Assign0bert/encoder/layer_9/attention/output/dense/biassave_1/RestoreV2:183*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias
?
save_1/Assign_184Assign2bert/encoder/layer_9/attention/output/dense/kernelsave_1/RestoreV2:184*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0
?
save_1/Assign_185Assign,bert/encoder/layer_9/attention/self/key/biassave_1/RestoreV2:185*
T0*?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_186Assign.bert/encoder/layer_9/attention/self/key/kernelsave_1/RestoreV2:186*
validate_shape(* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel
?
save_1/Assign_187Assign.bert/encoder/layer_9/attention/self/query/biassave_1/RestoreV2:187*A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_188Assign0bert/encoder/layer_9/attention/self/query/kernelsave_1/RestoreV2:188*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_189Assign.bert/encoder/layer_9/attention/self/value/biassave_1/RestoreV2:189*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias
?
save_1/Assign_190Assign0bert/encoder/layer_9/attention/self/value/kernelsave_1/RestoreV2:190*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_191Assign,bert/encoder/layer_9/intermediate/dense/biassave_1/RestoreV2:191*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_192Assign.bert/encoder/layer_9/intermediate/dense/kernelsave_1/RestoreV2:192* 
_output_shapes
:
??*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
validate_shape(
?
save_1/Assign_193Assign*bert/encoder/layer_9/output/LayerNorm/betasave_1/RestoreV2:193*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_194Assign+bert/encoder/layer_9/output/LayerNorm/gammasave_1/RestoreV2:194*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma
?
save_1/Assign_195Assign&bert/encoder/layer_9/output/dense/biassave_1/RestoreV2:195*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_196Assign(bert/encoder/layer_9/output/dense/kernelsave_1/RestoreV2:196*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_197Assignbert/pooler/dense/biassave_1/RestoreV2:197*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*)
_class
loc:@bert/pooler/dense/bias
?
save_1/Assign_198Assignbert/pooler/dense/kernelsave_1/RestoreV2:198*
use_locking(*
T0*+
_class!
loc:@bert/pooler/dense/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_199Assignoutput_biassave_1/RestoreV2:199*
use_locking(*
T0*
_class
loc:@output_bias*
validate_shape(*
_output_shapes
:
?
save_1/Assign_200Assignoutput_weightssave_1/RestoreV2:200*
use_locking(*
T0*!
_class
loc:@output_weights*
validate_shape(*
_output_shapes
:	?
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_100^save_1/Assign_101^save_1/Assign_102^save_1/Assign_103^save_1/Assign_104^save_1/Assign_105^save_1/Assign_106^save_1/Assign_107^save_1/Assign_108^save_1/Assign_109^save_1/Assign_11^save_1/Assign_110^save_1/Assign_111^save_1/Assign_112^save_1/Assign_113^save_1/Assign_114^save_1/Assign_115^save_1/Assign_116^save_1/Assign_117^save_1/Assign_118^save_1/Assign_119^save_1/Assign_12^save_1/Assign_120^save_1/Assign_121^save_1/Assign_122^save_1/Assign_123^save_1/Assign_124^save_1/Assign_125^save_1/Assign_126^save_1/Assign_127^save_1/Assign_128^save_1/Assign_129^save_1/Assign_13^save_1/Assign_130^save_1/Assign_131^save_1/Assign_132^save_1/Assign_133^save_1/Assign_134^save_1/Assign_135^save_1/Assign_136^save_1/Assign_137^save_1/Assign_138^save_1/Assign_139^save_1/Assign_14^save_1/Assign_140^save_1/Assign_141^save_1/Assign_142^save_1/Assign_143^save_1/Assign_144^save_1/Assign_145^save_1/Assign_146^save_1/Assign_147^save_1/Assign_148^save_1/Assign_149^save_1/Assign_15^save_1/Assign_150^save_1/Assign_151^save_1/Assign_152^save_1/Assign_153^save_1/Assign_154^save_1/Assign_155^save_1/Assign_156^save_1/Assign_157^save_1/Assign_158^save_1/Assign_159^save_1/Assign_16^save_1/Assign_160^save_1/Assign_161^save_1/Assign_162^save_1/Assign_163^save_1/Assign_164^save_1/Assign_165^save_1/Assign_166^save_1/Assign_167^save_1/Assign_168^save_1/Assign_169^save_1/Assign_17^save_1/Assign_170^save_1/Assign_171^save_1/Assign_172^save_1/Assign_173^save_1/Assign_174^save_1/Assign_175^save_1/Assign_176^save_1/Assign_177^save_1/Assign_178^save_1/Assign_179^save_1/Assign_18^save_1/Assign_180^save_1/Assign_181^save_1/Assign_182^save_1/Assign_183^save_1/Assign_184^save_1/Assign_185^save_1/Assign_186^save_1/Assign_187^save_1/Assign_188^save_1/Assign_189^save_1/Assign_19^save_1/Assign_190^save_1/Assign_191^save_1/Assign_192^save_1/Assign_193^save_1/Assign_194^save_1/Assign_195^save_1/Assign_196^save_1/Assign_197^save_1/Assign_198^save_1/Assign_199^save_1/Assign_2^save_1/Assign_20^save_1/Assign_200^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_8^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_84^save_1/Assign_85^save_1/Assign_86^save_1/Assign_87^save_1/Assign_88^save_1/Assign_89^save_1/Assign_9^save_1/Assign_90^save_1/Assign_91^save_1/Assign_92^save_1/Assign_93^save_1/Assign_94^save_1/Assign_95^save_1/Assign_96^save_1/Assign_97^save_1/Assign_98^save_1/Assign_99
1
save_1/restore_allNoOp^save_1/restore_shard "B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"?\
model_variables?\?\
?
 bert/embeddings/LayerNorm/beta:0%bert/embeddings/LayerNorm/beta/Assign%bert/embeddings/LayerNorm/beta/read:022bert/embeddings/LayerNorm/beta/Initializer/zeros:08
?
!bert/embeddings/LayerNorm/gamma:0&bert/embeddings/LayerNorm/gamma/Assign&bert/embeddings/LayerNorm/gamma/read:022bert/embeddings/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_0/attention/output/LayerNorm/beta:0;bert/encoder/layer_0/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_0/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_0/attention/output/LayerNorm/gamma:0<bert/encoder/layer_0/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_0/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_0/output/LayerNorm/beta:01bert/encoder/layer_0/output/LayerNorm/beta/Assign1bert/encoder/layer_0/output/LayerNorm/beta/read:02>bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_0/output/LayerNorm/gamma:02bert/encoder/layer_0/output/LayerNorm/gamma/Assign2bert/encoder/layer_0/output/LayerNorm/gamma/read:02>bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_1/attention/output/LayerNorm/beta:0;bert/encoder/layer_1/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_1/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_1/attention/output/LayerNorm/gamma:0<bert/encoder/layer_1/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_1/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_1/output/LayerNorm/beta:01bert/encoder/layer_1/output/LayerNorm/beta/Assign1bert/encoder/layer_1/output/LayerNorm/beta/read:02>bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_1/output/LayerNorm/gamma:02bert/encoder/layer_1/output/LayerNorm/gamma/Assign2bert/encoder/layer_1/output/LayerNorm/gamma/read:02>bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_2/attention/output/LayerNorm/beta:0;bert/encoder/layer_2/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_2/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_2/attention/output/LayerNorm/gamma:0<bert/encoder/layer_2/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_2/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_2/output/LayerNorm/beta:01bert/encoder/layer_2/output/LayerNorm/beta/Assign1bert/encoder/layer_2/output/LayerNorm/beta/read:02>bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_2/output/LayerNorm/gamma:02bert/encoder/layer_2/output/LayerNorm/gamma/Assign2bert/encoder/layer_2/output/LayerNorm/gamma/read:02>bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_3/attention/output/LayerNorm/beta:0;bert/encoder/layer_3/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_3/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_3/attention/output/LayerNorm/gamma:0<bert/encoder/layer_3/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_3/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_3/output/LayerNorm/beta:01bert/encoder/layer_3/output/LayerNorm/beta/Assign1bert/encoder/layer_3/output/LayerNorm/beta/read:02>bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_3/output/LayerNorm/gamma:02bert/encoder/layer_3/output/LayerNorm/gamma/Assign2bert/encoder/layer_3/output/LayerNorm/gamma/read:02>bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_4/attention/output/LayerNorm/beta:0;bert/encoder/layer_4/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_4/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_4/attention/output/LayerNorm/gamma:0<bert/encoder/layer_4/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_4/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_4/output/LayerNorm/beta:01bert/encoder/layer_4/output/LayerNorm/beta/Assign1bert/encoder/layer_4/output/LayerNorm/beta/read:02>bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_4/output/LayerNorm/gamma:02bert/encoder/layer_4/output/LayerNorm/gamma/Assign2bert/encoder/layer_4/output/LayerNorm/gamma/read:02>bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_5/attention/output/LayerNorm/beta:0;bert/encoder/layer_5/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_5/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_5/attention/output/LayerNorm/gamma:0<bert/encoder/layer_5/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_5/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_5/output/LayerNorm/beta:01bert/encoder/layer_5/output/LayerNorm/beta/Assign1bert/encoder/layer_5/output/LayerNorm/beta/read:02>bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_5/output/LayerNorm/gamma:02bert/encoder/layer_5/output/LayerNorm/gamma/Assign2bert/encoder/layer_5/output/LayerNorm/gamma/read:02>bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_6/attention/output/LayerNorm/beta:0;bert/encoder/layer_6/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_6/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_6/attention/output/LayerNorm/gamma:0<bert/encoder/layer_6/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_6/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_6/output/LayerNorm/beta:01bert/encoder/layer_6/output/LayerNorm/beta/Assign1bert/encoder/layer_6/output/LayerNorm/beta/read:02>bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_6/output/LayerNorm/gamma:02bert/encoder/layer_6/output/LayerNorm/gamma/Assign2bert/encoder/layer_6/output/LayerNorm/gamma/read:02>bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_7/attention/output/LayerNorm/beta:0;bert/encoder/layer_7/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_7/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_7/attention/output/LayerNorm/gamma:0<bert/encoder/layer_7/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_7/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_7/output/LayerNorm/beta:01bert/encoder/layer_7/output/LayerNorm/beta/Assign1bert/encoder/layer_7/output/LayerNorm/beta/read:02>bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_7/output/LayerNorm/gamma:02bert/encoder/layer_7/output/LayerNorm/gamma/Assign2bert/encoder/layer_7/output/LayerNorm/gamma/read:02>bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_8/attention/output/LayerNorm/beta:0;bert/encoder/layer_8/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_8/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_8/attention/output/LayerNorm/gamma:0<bert/encoder/layer_8/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_8/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_8/output/LayerNorm/beta:01bert/encoder/layer_8/output/LayerNorm/beta/Assign1bert/encoder/layer_8/output/LayerNorm/beta/read:02>bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_8/output/LayerNorm/gamma:02bert/encoder/layer_8/output/LayerNorm/gamma/Assign2bert/encoder/layer_8/output/LayerNorm/gamma/read:02>bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/ones:08
?
6bert/encoder/layer_9/attention/output/LayerNorm/beta:0;bert/encoder/layer_9/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_9/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_9/attention/output/LayerNorm/gamma:0<bert/encoder/layer_9/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_9/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/ones:08
?
,bert/encoder/layer_9/output/LayerNorm/beta:01bert/encoder/layer_9/output/LayerNorm/beta/Assign1bert/encoder/layer_9/output/LayerNorm/beta/read:02>bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_9/output/LayerNorm/gamma:02bert/encoder/layer_9/output/LayerNorm/gamma/Assign2bert/encoder/layer_9/output/LayerNorm/gamma/read:02>bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/ones:08
?
7bert/encoder/layer_10/attention/output/LayerNorm/beta:0<bert/encoder/layer_10/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_10/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zeros:08
?
8bert/encoder/layer_10/attention/output/LayerNorm/gamma:0=bert/encoder/layer_10/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_10/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/ones:08
?
-bert/encoder/layer_10/output/LayerNorm/beta:02bert/encoder/layer_10/output/LayerNorm/beta/Assign2bert/encoder/layer_10/output/LayerNorm/beta/read:02?bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zeros:08
?
.bert/encoder/layer_10/output/LayerNorm/gamma:03bert/encoder/layer_10/output/LayerNorm/gamma/Assign3bert/encoder/layer_10/output/LayerNorm/gamma/read:02?bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/ones:08
?
7bert/encoder/layer_11/attention/output/LayerNorm/beta:0<bert/encoder/layer_11/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_11/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zeros:08
?
8bert/encoder/layer_11/attention/output/LayerNorm/gamma:0=bert/encoder/layer_11/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_11/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/ones:08
?
-bert/encoder/layer_11/output/LayerNorm/beta:02bert/encoder/layer_11/output/LayerNorm/beta/Assign2bert/encoder/layer_11/output/LayerNorm/beta/read:02?bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zeros:08
?
.bert/encoder/layer_11/output/LayerNorm/gamma:03bert/encoder/layer_11/output/LayerNorm/gamma/Assign3bert/encoder/layer_11/output/LayerNorm/gamma/read:02?bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/ones:08"-
losses#
!
loss/mean_squared_error/value:0"??
trainable_variables????
?
!bert/embeddings/word_embeddings:0&bert/embeddings/word_embeddings/Assign&bert/embeddings/word_embeddings/read:02>bert/embeddings/word_embeddings/Initializer/truncated_normal:08
?
'bert/embeddings/token_type_embeddings:0,bert/embeddings/token_type_embeddings/Assign,bert/embeddings/token_type_embeddings/read:02Dbert/embeddings/token_type_embeddings/Initializer/truncated_normal:08
?
%bert/embeddings/position_embeddings:0*bert/embeddings/position_embeddings/Assign*bert/embeddings/position_embeddings/read:02Bbert/embeddings/position_embeddings/Initializer/truncated_normal:08
?
 bert/embeddings/LayerNorm/beta:0%bert/embeddings/LayerNorm/beta/Assign%bert/embeddings/LayerNorm/beta/read:022bert/embeddings/LayerNorm/beta/Initializer/zeros:08
?
!bert/embeddings/LayerNorm/gamma:0&bert/embeddings/LayerNorm/gamma/Assign&bert/embeddings/LayerNorm/gamma/read:022bert/embeddings/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_0/attention/self/query/kernel:07bert/encoder/layer_0/attention/self/query/kernel/Assign7bert/encoder/layer_0/attention/self/query/kernel/read:02Obert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_0/attention/self/query/bias:05bert/encoder/layer_0/attention/self/query/bias/Assign5bert/encoder/layer_0/attention/self/query/bias/read:02Bbert/encoder/layer_0/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_0/attention/self/key/kernel:05bert/encoder/layer_0/attention/self/key/kernel/Assign5bert/encoder/layer_0/attention/self/key/kernel/read:02Mbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_0/attention/self/key/bias:03bert/encoder/layer_0/attention/self/key/bias/Assign3bert/encoder/layer_0/attention/self/key/bias/read:02@bert/encoder/layer_0/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_0/attention/self/value/kernel:07bert/encoder/layer_0/attention/self/value/kernel/Assign7bert/encoder/layer_0/attention/self/value/kernel/read:02Obert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_0/attention/self/value/bias:05bert/encoder/layer_0/attention/self/value/bias/Assign5bert/encoder/layer_0/attention/self/value/bias/read:02Bbert/encoder/layer_0/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_0/attention/output/dense/kernel:09bert/encoder/layer_0/attention/output/dense/kernel/Assign9bert/encoder/layer_0/attention/output/dense/kernel/read:02Qbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_0/attention/output/dense/bias:07bert/encoder/layer_0/attention/output/dense/bias/Assign7bert/encoder/layer_0/attention/output/dense/bias/read:02Dbert/encoder/layer_0/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_0/attention/output/LayerNorm/beta:0;bert/encoder/layer_0/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_0/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_0/attention/output/LayerNorm/gamma:0<bert/encoder/layer_0/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_0/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_0/intermediate/dense/kernel:05bert/encoder/layer_0/intermediate/dense/kernel/Assign5bert/encoder/layer_0/intermediate/dense/kernel/read:02Mbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_0/intermediate/dense/bias:03bert/encoder/layer_0/intermediate/dense/bias/Assign3bert/encoder/layer_0/intermediate/dense/bias/read:02@bert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_0/output/dense/kernel:0/bert/encoder/layer_0/output/dense/kernel/Assign/bert/encoder/layer_0/output/dense/kernel/read:02Gbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_0/output/dense/bias:0-bert/encoder/layer_0/output/dense/bias/Assign-bert/encoder/layer_0/output/dense/bias/read:02:bert/encoder/layer_0/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_0/output/LayerNorm/beta:01bert/encoder/layer_0/output/LayerNorm/beta/Assign1bert/encoder/layer_0/output/LayerNorm/beta/read:02>bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_0/output/LayerNorm/gamma:02bert/encoder/layer_0/output/LayerNorm/gamma/Assign2bert/encoder/layer_0/output/LayerNorm/gamma/read:02>bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_1/attention/self/query/kernel:07bert/encoder/layer_1/attention/self/query/kernel/Assign7bert/encoder/layer_1/attention/self/query/kernel/read:02Obert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_1/attention/self/query/bias:05bert/encoder/layer_1/attention/self/query/bias/Assign5bert/encoder/layer_1/attention/self/query/bias/read:02Bbert/encoder/layer_1/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_1/attention/self/key/kernel:05bert/encoder/layer_1/attention/self/key/kernel/Assign5bert/encoder/layer_1/attention/self/key/kernel/read:02Mbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_1/attention/self/key/bias:03bert/encoder/layer_1/attention/self/key/bias/Assign3bert/encoder/layer_1/attention/self/key/bias/read:02@bert/encoder/layer_1/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_1/attention/self/value/kernel:07bert/encoder/layer_1/attention/self/value/kernel/Assign7bert/encoder/layer_1/attention/self/value/kernel/read:02Obert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_1/attention/self/value/bias:05bert/encoder/layer_1/attention/self/value/bias/Assign5bert/encoder/layer_1/attention/self/value/bias/read:02Bbert/encoder/layer_1/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_1/attention/output/dense/kernel:09bert/encoder/layer_1/attention/output/dense/kernel/Assign9bert/encoder/layer_1/attention/output/dense/kernel/read:02Qbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_1/attention/output/dense/bias:07bert/encoder/layer_1/attention/output/dense/bias/Assign7bert/encoder/layer_1/attention/output/dense/bias/read:02Dbert/encoder/layer_1/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_1/attention/output/LayerNorm/beta:0;bert/encoder/layer_1/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_1/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_1/attention/output/LayerNorm/gamma:0<bert/encoder/layer_1/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_1/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_1/intermediate/dense/kernel:05bert/encoder/layer_1/intermediate/dense/kernel/Assign5bert/encoder/layer_1/intermediate/dense/kernel/read:02Mbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_1/intermediate/dense/bias:03bert/encoder/layer_1/intermediate/dense/bias/Assign3bert/encoder/layer_1/intermediate/dense/bias/read:02@bert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_1/output/dense/kernel:0/bert/encoder/layer_1/output/dense/kernel/Assign/bert/encoder/layer_1/output/dense/kernel/read:02Gbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_1/output/dense/bias:0-bert/encoder/layer_1/output/dense/bias/Assign-bert/encoder/layer_1/output/dense/bias/read:02:bert/encoder/layer_1/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_1/output/LayerNorm/beta:01bert/encoder/layer_1/output/LayerNorm/beta/Assign1bert/encoder/layer_1/output/LayerNorm/beta/read:02>bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_1/output/LayerNorm/gamma:02bert/encoder/layer_1/output/LayerNorm/gamma/Assign2bert/encoder/layer_1/output/LayerNorm/gamma/read:02>bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_2/attention/self/query/kernel:07bert/encoder/layer_2/attention/self/query/kernel/Assign7bert/encoder/layer_2/attention/self/query/kernel/read:02Obert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_2/attention/self/query/bias:05bert/encoder/layer_2/attention/self/query/bias/Assign5bert/encoder/layer_2/attention/self/query/bias/read:02Bbert/encoder/layer_2/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_2/attention/self/key/kernel:05bert/encoder/layer_2/attention/self/key/kernel/Assign5bert/encoder/layer_2/attention/self/key/kernel/read:02Mbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_2/attention/self/key/bias:03bert/encoder/layer_2/attention/self/key/bias/Assign3bert/encoder/layer_2/attention/self/key/bias/read:02@bert/encoder/layer_2/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_2/attention/self/value/kernel:07bert/encoder/layer_2/attention/self/value/kernel/Assign7bert/encoder/layer_2/attention/self/value/kernel/read:02Obert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_2/attention/self/value/bias:05bert/encoder/layer_2/attention/self/value/bias/Assign5bert/encoder/layer_2/attention/self/value/bias/read:02Bbert/encoder/layer_2/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_2/attention/output/dense/kernel:09bert/encoder/layer_2/attention/output/dense/kernel/Assign9bert/encoder/layer_2/attention/output/dense/kernel/read:02Qbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_2/attention/output/dense/bias:07bert/encoder/layer_2/attention/output/dense/bias/Assign7bert/encoder/layer_2/attention/output/dense/bias/read:02Dbert/encoder/layer_2/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_2/attention/output/LayerNorm/beta:0;bert/encoder/layer_2/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_2/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_2/attention/output/LayerNorm/gamma:0<bert/encoder/layer_2/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_2/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_2/intermediate/dense/kernel:05bert/encoder/layer_2/intermediate/dense/kernel/Assign5bert/encoder/layer_2/intermediate/dense/kernel/read:02Mbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_2/intermediate/dense/bias:03bert/encoder/layer_2/intermediate/dense/bias/Assign3bert/encoder/layer_2/intermediate/dense/bias/read:02@bert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_2/output/dense/kernel:0/bert/encoder/layer_2/output/dense/kernel/Assign/bert/encoder/layer_2/output/dense/kernel/read:02Gbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_2/output/dense/bias:0-bert/encoder/layer_2/output/dense/bias/Assign-bert/encoder/layer_2/output/dense/bias/read:02:bert/encoder/layer_2/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_2/output/LayerNorm/beta:01bert/encoder/layer_2/output/LayerNorm/beta/Assign1bert/encoder/layer_2/output/LayerNorm/beta/read:02>bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_2/output/LayerNorm/gamma:02bert/encoder/layer_2/output/LayerNorm/gamma/Assign2bert/encoder/layer_2/output/LayerNorm/gamma/read:02>bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_3/attention/self/query/kernel:07bert/encoder/layer_3/attention/self/query/kernel/Assign7bert/encoder/layer_3/attention/self/query/kernel/read:02Obert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_3/attention/self/query/bias:05bert/encoder/layer_3/attention/self/query/bias/Assign5bert/encoder/layer_3/attention/self/query/bias/read:02Bbert/encoder/layer_3/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_3/attention/self/key/kernel:05bert/encoder/layer_3/attention/self/key/kernel/Assign5bert/encoder/layer_3/attention/self/key/kernel/read:02Mbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_3/attention/self/key/bias:03bert/encoder/layer_3/attention/self/key/bias/Assign3bert/encoder/layer_3/attention/self/key/bias/read:02@bert/encoder/layer_3/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_3/attention/self/value/kernel:07bert/encoder/layer_3/attention/self/value/kernel/Assign7bert/encoder/layer_3/attention/self/value/kernel/read:02Obert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_3/attention/self/value/bias:05bert/encoder/layer_3/attention/self/value/bias/Assign5bert/encoder/layer_3/attention/self/value/bias/read:02Bbert/encoder/layer_3/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_3/attention/output/dense/kernel:09bert/encoder/layer_3/attention/output/dense/kernel/Assign9bert/encoder/layer_3/attention/output/dense/kernel/read:02Qbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_3/attention/output/dense/bias:07bert/encoder/layer_3/attention/output/dense/bias/Assign7bert/encoder/layer_3/attention/output/dense/bias/read:02Dbert/encoder/layer_3/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_3/attention/output/LayerNorm/beta:0;bert/encoder/layer_3/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_3/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_3/attention/output/LayerNorm/gamma:0<bert/encoder/layer_3/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_3/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_3/intermediate/dense/kernel:05bert/encoder/layer_3/intermediate/dense/kernel/Assign5bert/encoder/layer_3/intermediate/dense/kernel/read:02Mbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_3/intermediate/dense/bias:03bert/encoder/layer_3/intermediate/dense/bias/Assign3bert/encoder/layer_3/intermediate/dense/bias/read:02@bert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_3/output/dense/kernel:0/bert/encoder/layer_3/output/dense/kernel/Assign/bert/encoder/layer_3/output/dense/kernel/read:02Gbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_3/output/dense/bias:0-bert/encoder/layer_3/output/dense/bias/Assign-bert/encoder/layer_3/output/dense/bias/read:02:bert/encoder/layer_3/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_3/output/LayerNorm/beta:01bert/encoder/layer_3/output/LayerNorm/beta/Assign1bert/encoder/layer_3/output/LayerNorm/beta/read:02>bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_3/output/LayerNorm/gamma:02bert/encoder/layer_3/output/LayerNorm/gamma/Assign2bert/encoder/layer_3/output/LayerNorm/gamma/read:02>bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_4/attention/self/query/kernel:07bert/encoder/layer_4/attention/self/query/kernel/Assign7bert/encoder/layer_4/attention/self/query/kernel/read:02Obert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_4/attention/self/query/bias:05bert/encoder/layer_4/attention/self/query/bias/Assign5bert/encoder/layer_4/attention/self/query/bias/read:02Bbert/encoder/layer_4/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_4/attention/self/key/kernel:05bert/encoder/layer_4/attention/self/key/kernel/Assign5bert/encoder/layer_4/attention/self/key/kernel/read:02Mbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_4/attention/self/key/bias:03bert/encoder/layer_4/attention/self/key/bias/Assign3bert/encoder/layer_4/attention/self/key/bias/read:02@bert/encoder/layer_4/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_4/attention/self/value/kernel:07bert/encoder/layer_4/attention/self/value/kernel/Assign7bert/encoder/layer_4/attention/self/value/kernel/read:02Obert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_4/attention/self/value/bias:05bert/encoder/layer_4/attention/self/value/bias/Assign5bert/encoder/layer_4/attention/self/value/bias/read:02Bbert/encoder/layer_4/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_4/attention/output/dense/kernel:09bert/encoder/layer_4/attention/output/dense/kernel/Assign9bert/encoder/layer_4/attention/output/dense/kernel/read:02Qbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_4/attention/output/dense/bias:07bert/encoder/layer_4/attention/output/dense/bias/Assign7bert/encoder/layer_4/attention/output/dense/bias/read:02Dbert/encoder/layer_4/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_4/attention/output/LayerNorm/beta:0;bert/encoder/layer_4/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_4/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_4/attention/output/LayerNorm/gamma:0<bert/encoder/layer_4/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_4/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_4/intermediate/dense/kernel:05bert/encoder/layer_4/intermediate/dense/kernel/Assign5bert/encoder/layer_4/intermediate/dense/kernel/read:02Mbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_4/intermediate/dense/bias:03bert/encoder/layer_4/intermediate/dense/bias/Assign3bert/encoder/layer_4/intermediate/dense/bias/read:02@bert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_4/output/dense/kernel:0/bert/encoder/layer_4/output/dense/kernel/Assign/bert/encoder/layer_4/output/dense/kernel/read:02Gbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_4/output/dense/bias:0-bert/encoder/layer_4/output/dense/bias/Assign-bert/encoder/layer_4/output/dense/bias/read:02:bert/encoder/layer_4/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_4/output/LayerNorm/beta:01bert/encoder/layer_4/output/LayerNorm/beta/Assign1bert/encoder/layer_4/output/LayerNorm/beta/read:02>bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_4/output/LayerNorm/gamma:02bert/encoder/layer_4/output/LayerNorm/gamma/Assign2bert/encoder/layer_4/output/LayerNorm/gamma/read:02>bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_5/attention/self/query/kernel:07bert/encoder/layer_5/attention/self/query/kernel/Assign7bert/encoder/layer_5/attention/self/query/kernel/read:02Obert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_5/attention/self/query/bias:05bert/encoder/layer_5/attention/self/query/bias/Assign5bert/encoder/layer_5/attention/self/query/bias/read:02Bbert/encoder/layer_5/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_5/attention/self/key/kernel:05bert/encoder/layer_5/attention/self/key/kernel/Assign5bert/encoder/layer_5/attention/self/key/kernel/read:02Mbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_5/attention/self/key/bias:03bert/encoder/layer_5/attention/self/key/bias/Assign3bert/encoder/layer_5/attention/self/key/bias/read:02@bert/encoder/layer_5/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_5/attention/self/value/kernel:07bert/encoder/layer_5/attention/self/value/kernel/Assign7bert/encoder/layer_5/attention/self/value/kernel/read:02Obert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_5/attention/self/value/bias:05bert/encoder/layer_5/attention/self/value/bias/Assign5bert/encoder/layer_5/attention/self/value/bias/read:02Bbert/encoder/layer_5/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_5/attention/output/dense/kernel:09bert/encoder/layer_5/attention/output/dense/kernel/Assign9bert/encoder/layer_5/attention/output/dense/kernel/read:02Qbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_5/attention/output/dense/bias:07bert/encoder/layer_5/attention/output/dense/bias/Assign7bert/encoder/layer_5/attention/output/dense/bias/read:02Dbert/encoder/layer_5/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_5/attention/output/LayerNorm/beta:0;bert/encoder/layer_5/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_5/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_5/attention/output/LayerNorm/gamma:0<bert/encoder/layer_5/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_5/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_5/intermediate/dense/kernel:05bert/encoder/layer_5/intermediate/dense/kernel/Assign5bert/encoder/layer_5/intermediate/dense/kernel/read:02Mbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_5/intermediate/dense/bias:03bert/encoder/layer_5/intermediate/dense/bias/Assign3bert/encoder/layer_5/intermediate/dense/bias/read:02@bert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_5/output/dense/kernel:0/bert/encoder/layer_5/output/dense/kernel/Assign/bert/encoder/layer_5/output/dense/kernel/read:02Gbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_5/output/dense/bias:0-bert/encoder/layer_5/output/dense/bias/Assign-bert/encoder/layer_5/output/dense/bias/read:02:bert/encoder/layer_5/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_5/output/LayerNorm/beta:01bert/encoder/layer_5/output/LayerNorm/beta/Assign1bert/encoder/layer_5/output/LayerNorm/beta/read:02>bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_5/output/LayerNorm/gamma:02bert/encoder/layer_5/output/LayerNorm/gamma/Assign2bert/encoder/layer_5/output/LayerNorm/gamma/read:02>bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_6/attention/self/query/kernel:07bert/encoder/layer_6/attention/self/query/kernel/Assign7bert/encoder/layer_6/attention/self/query/kernel/read:02Obert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_6/attention/self/query/bias:05bert/encoder/layer_6/attention/self/query/bias/Assign5bert/encoder/layer_6/attention/self/query/bias/read:02Bbert/encoder/layer_6/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_6/attention/self/key/kernel:05bert/encoder/layer_6/attention/self/key/kernel/Assign5bert/encoder/layer_6/attention/self/key/kernel/read:02Mbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_6/attention/self/key/bias:03bert/encoder/layer_6/attention/self/key/bias/Assign3bert/encoder/layer_6/attention/self/key/bias/read:02@bert/encoder/layer_6/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_6/attention/self/value/kernel:07bert/encoder/layer_6/attention/self/value/kernel/Assign7bert/encoder/layer_6/attention/self/value/kernel/read:02Obert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_6/attention/self/value/bias:05bert/encoder/layer_6/attention/self/value/bias/Assign5bert/encoder/layer_6/attention/self/value/bias/read:02Bbert/encoder/layer_6/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_6/attention/output/dense/kernel:09bert/encoder/layer_6/attention/output/dense/kernel/Assign9bert/encoder/layer_6/attention/output/dense/kernel/read:02Qbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_6/attention/output/dense/bias:07bert/encoder/layer_6/attention/output/dense/bias/Assign7bert/encoder/layer_6/attention/output/dense/bias/read:02Dbert/encoder/layer_6/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_6/attention/output/LayerNorm/beta:0;bert/encoder/layer_6/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_6/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_6/attention/output/LayerNorm/gamma:0<bert/encoder/layer_6/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_6/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_6/intermediate/dense/kernel:05bert/encoder/layer_6/intermediate/dense/kernel/Assign5bert/encoder/layer_6/intermediate/dense/kernel/read:02Mbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_6/intermediate/dense/bias:03bert/encoder/layer_6/intermediate/dense/bias/Assign3bert/encoder/layer_6/intermediate/dense/bias/read:02@bert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_6/output/dense/kernel:0/bert/encoder/layer_6/output/dense/kernel/Assign/bert/encoder/layer_6/output/dense/kernel/read:02Gbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_6/output/dense/bias:0-bert/encoder/layer_6/output/dense/bias/Assign-bert/encoder/layer_6/output/dense/bias/read:02:bert/encoder/layer_6/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_6/output/LayerNorm/beta:01bert/encoder/layer_6/output/LayerNorm/beta/Assign1bert/encoder/layer_6/output/LayerNorm/beta/read:02>bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_6/output/LayerNorm/gamma:02bert/encoder/layer_6/output/LayerNorm/gamma/Assign2bert/encoder/layer_6/output/LayerNorm/gamma/read:02>bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_7/attention/self/query/kernel:07bert/encoder/layer_7/attention/self/query/kernel/Assign7bert/encoder/layer_7/attention/self/query/kernel/read:02Obert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_7/attention/self/query/bias:05bert/encoder/layer_7/attention/self/query/bias/Assign5bert/encoder/layer_7/attention/self/query/bias/read:02Bbert/encoder/layer_7/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_7/attention/self/key/kernel:05bert/encoder/layer_7/attention/self/key/kernel/Assign5bert/encoder/layer_7/attention/self/key/kernel/read:02Mbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_7/attention/self/key/bias:03bert/encoder/layer_7/attention/self/key/bias/Assign3bert/encoder/layer_7/attention/self/key/bias/read:02@bert/encoder/layer_7/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_7/attention/self/value/kernel:07bert/encoder/layer_7/attention/self/value/kernel/Assign7bert/encoder/layer_7/attention/self/value/kernel/read:02Obert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_7/attention/self/value/bias:05bert/encoder/layer_7/attention/self/value/bias/Assign5bert/encoder/layer_7/attention/self/value/bias/read:02Bbert/encoder/layer_7/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_7/attention/output/dense/kernel:09bert/encoder/layer_7/attention/output/dense/kernel/Assign9bert/encoder/layer_7/attention/output/dense/kernel/read:02Qbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_7/attention/output/dense/bias:07bert/encoder/layer_7/attention/output/dense/bias/Assign7bert/encoder/layer_7/attention/output/dense/bias/read:02Dbert/encoder/layer_7/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_7/attention/output/LayerNorm/beta:0;bert/encoder/layer_7/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_7/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_7/attention/output/LayerNorm/gamma:0<bert/encoder/layer_7/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_7/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_7/intermediate/dense/kernel:05bert/encoder/layer_7/intermediate/dense/kernel/Assign5bert/encoder/layer_7/intermediate/dense/kernel/read:02Mbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_7/intermediate/dense/bias:03bert/encoder/layer_7/intermediate/dense/bias/Assign3bert/encoder/layer_7/intermediate/dense/bias/read:02@bert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_7/output/dense/kernel:0/bert/encoder/layer_7/output/dense/kernel/Assign/bert/encoder/layer_7/output/dense/kernel/read:02Gbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_7/output/dense/bias:0-bert/encoder/layer_7/output/dense/bias/Assign-bert/encoder/layer_7/output/dense/bias/read:02:bert/encoder/layer_7/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_7/output/LayerNorm/beta:01bert/encoder/layer_7/output/LayerNorm/beta/Assign1bert/encoder/layer_7/output/LayerNorm/beta/read:02>bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_7/output/LayerNorm/gamma:02bert/encoder/layer_7/output/LayerNorm/gamma/Assign2bert/encoder/layer_7/output/LayerNorm/gamma/read:02>bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_8/attention/self/query/kernel:07bert/encoder/layer_8/attention/self/query/kernel/Assign7bert/encoder/layer_8/attention/self/query/kernel/read:02Obert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_8/attention/self/query/bias:05bert/encoder/layer_8/attention/self/query/bias/Assign5bert/encoder/layer_8/attention/self/query/bias/read:02Bbert/encoder/layer_8/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_8/attention/self/key/kernel:05bert/encoder/layer_8/attention/self/key/kernel/Assign5bert/encoder/layer_8/attention/self/key/kernel/read:02Mbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_8/attention/self/key/bias:03bert/encoder/layer_8/attention/self/key/bias/Assign3bert/encoder/layer_8/attention/self/key/bias/read:02@bert/encoder/layer_8/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_8/attention/self/value/kernel:07bert/encoder/layer_8/attention/self/value/kernel/Assign7bert/encoder/layer_8/attention/self/value/kernel/read:02Obert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_8/attention/self/value/bias:05bert/encoder/layer_8/attention/self/value/bias/Assign5bert/encoder/layer_8/attention/self/value/bias/read:02Bbert/encoder/layer_8/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_8/attention/output/dense/kernel:09bert/encoder/layer_8/attention/output/dense/kernel/Assign9bert/encoder/layer_8/attention/output/dense/kernel/read:02Qbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_8/attention/output/dense/bias:07bert/encoder/layer_8/attention/output/dense/bias/Assign7bert/encoder/layer_8/attention/output/dense/bias/read:02Dbert/encoder/layer_8/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_8/attention/output/LayerNorm/beta:0;bert/encoder/layer_8/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_8/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_8/attention/output/LayerNorm/gamma:0<bert/encoder/layer_8/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_8/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_8/intermediate/dense/kernel:05bert/encoder/layer_8/intermediate/dense/kernel/Assign5bert/encoder/layer_8/intermediate/dense/kernel/read:02Mbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_8/intermediate/dense/bias:03bert/encoder/layer_8/intermediate/dense/bias/Assign3bert/encoder/layer_8/intermediate/dense/bias/read:02@bert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_8/output/dense/kernel:0/bert/encoder/layer_8/output/dense/kernel/Assign/bert/encoder/layer_8/output/dense/kernel/read:02Gbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_8/output/dense/bias:0-bert/encoder/layer_8/output/dense/bias/Assign-bert/encoder/layer_8/output/dense/bias/read:02:bert/encoder/layer_8/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_8/output/LayerNorm/beta:01bert/encoder/layer_8/output/LayerNorm/beta/Assign1bert/encoder/layer_8/output/LayerNorm/beta/read:02>bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_8/output/LayerNorm/gamma:02bert/encoder/layer_8/output/LayerNorm/gamma/Assign2bert/encoder/layer_8/output/LayerNorm/gamma/read:02>bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_9/attention/self/query/kernel:07bert/encoder/layer_9/attention/self/query/kernel/Assign7bert/encoder/layer_9/attention/self/query/kernel/read:02Obert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_9/attention/self/query/bias:05bert/encoder/layer_9/attention/self/query/bias/Assign5bert/encoder/layer_9/attention/self/query/bias/read:02Bbert/encoder/layer_9/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_9/attention/self/key/kernel:05bert/encoder/layer_9/attention/self/key/kernel/Assign5bert/encoder/layer_9/attention/self/key/kernel/read:02Mbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_9/attention/self/key/bias:03bert/encoder/layer_9/attention/self/key/bias/Assign3bert/encoder/layer_9/attention/self/key/bias/read:02@bert/encoder/layer_9/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_9/attention/self/value/kernel:07bert/encoder/layer_9/attention/self/value/kernel/Assign7bert/encoder/layer_9/attention/self/value/kernel/read:02Obert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_9/attention/self/value/bias:05bert/encoder/layer_9/attention/self/value/bias/Assign5bert/encoder/layer_9/attention/self/value/bias/read:02Bbert/encoder/layer_9/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_9/attention/output/dense/kernel:09bert/encoder/layer_9/attention/output/dense/kernel/Assign9bert/encoder/layer_9/attention/output/dense/kernel/read:02Qbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_9/attention/output/dense/bias:07bert/encoder/layer_9/attention/output/dense/bias/Assign7bert/encoder/layer_9/attention/output/dense/bias/read:02Dbert/encoder/layer_9/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_9/attention/output/LayerNorm/beta:0;bert/encoder/layer_9/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_9/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_9/attention/output/LayerNorm/gamma:0<bert/encoder/layer_9/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_9/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_9/intermediate/dense/kernel:05bert/encoder/layer_9/intermediate/dense/kernel/Assign5bert/encoder/layer_9/intermediate/dense/kernel/read:02Mbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_9/intermediate/dense/bias:03bert/encoder/layer_9/intermediate/dense/bias/Assign3bert/encoder/layer_9/intermediate/dense/bias/read:02@bert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_9/output/dense/kernel:0/bert/encoder/layer_9/output/dense/kernel/Assign/bert/encoder/layer_9/output/dense/kernel/read:02Gbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_9/output/dense/bias:0-bert/encoder/layer_9/output/dense/bias/Assign-bert/encoder/layer_9/output/dense/bias/read:02:bert/encoder/layer_9/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_9/output/LayerNorm/beta:01bert/encoder/layer_9/output/LayerNorm/beta/Assign1bert/encoder/layer_9/output/LayerNorm/beta/read:02>bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_9/output/LayerNorm/gamma:02bert/encoder/layer_9/output/LayerNorm/gamma/Assign2bert/encoder/layer_9/output/LayerNorm/gamma/read:02>bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/ones:08
?
3bert/encoder/layer_10/attention/self/query/kernel:08bert/encoder/layer_10/attention/self/query/kernel/Assign8bert/encoder/layer_10/attention/self/query/kernel/read:02Pbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal:08
?
1bert/encoder/layer_10/attention/self/query/bias:06bert/encoder/layer_10/attention/self/query/bias/Assign6bert/encoder/layer_10/attention/self/query/bias/read:02Cbert/encoder/layer_10/attention/self/query/bias/Initializer/zeros:08
?
1bert/encoder/layer_10/attention/self/key/kernel:06bert/encoder/layer_10/attention/self/key/kernel/Assign6bert/encoder/layer_10/attention/self/key/kernel/read:02Nbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal:08
?
/bert/encoder/layer_10/attention/self/key/bias:04bert/encoder/layer_10/attention/self/key/bias/Assign4bert/encoder/layer_10/attention/self/key/bias/read:02Abert/encoder/layer_10/attention/self/key/bias/Initializer/zeros:08
?
3bert/encoder/layer_10/attention/self/value/kernel:08bert/encoder/layer_10/attention/self/value/kernel/Assign8bert/encoder/layer_10/attention/self/value/kernel/read:02Pbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal:08
?
1bert/encoder/layer_10/attention/self/value/bias:06bert/encoder/layer_10/attention/self/value/bias/Assign6bert/encoder/layer_10/attention/self/value/bias/read:02Cbert/encoder/layer_10/attention/self/value/bias/Initializer/zeros:08
?
5bert/encoder/layer_10/attention/output/dense/kernel:0:bert/encoder/layer_10/attention/output/dense/kernel/Assign:bert/encoder/layer_10/attention/output/dense/kernel/read:02Rbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal:08
?
3bert/encoder/layer_10/attention/output/dense/bias:08bert/encoder/layer_10/attention/output/dense/bias/Assign8bert/encoder/layer_10/attention/output/dense/bias/read:02Ebert/encoder/layer_10/attention/output/dense/bias/Initializer/zeros:08
?
7bert/encoder/layer_10/attention/output/LayerNorm/beta:0<bert/encoder/layer_10/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_10/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zeros:08
?
8bert/encoder/layer_10/attention/output/LayerNorm/gamma:0=bert/encoder/layer_10/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_10/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/ones:08
?
1bert/encoder/layer_10/intermediate/dense/kernel:06bert/encoder/layer_10/intermediate/dense/kernel/Assign6bert/encoder/layer_10/intermediate/dense/kernel/read:02Nbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal:08
?
/bert/encoder/layer_10/intermediate/dense/bias:04bert/encoder/layer_10/intermediate/dense/bias/Assign4bert/encoder/layer_10/intermediate/dense/bias/read:02Abert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros:08
?
+bert/encoder/layer_10/output/dense/kernel:00bert/encoder/layer_10/output/dense/kernel/Assign0bert/encoder/layer_10/output/dense/kernel/read:02Hbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal:08
?
)bert/encoder/layer_10/output/dense/bias:0.bert/encoder/layer_10/output/dense/bias/Assign.bert/encoder/layer_10/output/dense/bias/read:02;bert/encoder/layer_10/output/dense/bias/Initializer/zeros:08
?
-bert/encoder/layer_10/output/LayerNorm/beta:02bert/encoder/layer_10/output/LayerNorm/beta/Assign2bert/encoder/layer_10/output/LayerNorm/beta/read:02?bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zeros:08
?
.bert/encoder/layer_10/output/LayerNorm/gamma:03bert/encoder/layer_10/output/LayerNorm/gamma/Assign3bert/encoder/layer_10/output/LayerNorm/gamma/read:02?bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/ones:08
?
3bert/encoder/layer_11/attention/self/query/kernel:08bert/encoder/layer_11/attention/self/query/kernel/Assign8bert/encoder/layer_11/attention/self/query/kernel/read:02Pbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal:08
?
1bert/encoder/layer_11/attention/self/query/bias:06bert/encoder/layer_11/attention/self/query/bias/Assign6bert/encoder/layer_11/attention/self/query/bias/read:02Cbert/encoder/layer_11/attention/self/query/bias/Initializer/zeros:08
?
1bert/encoder/layer_11/attention/self/key/kernel:06bert/encoder/layer_11/attention/self/key/kernel/Assign6bert/encoder/layer_11/attention/self/key/kernel/read:02Nbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal:08
?
/bert/encoder/layer_11/attention/self/key/bias:04bert/encoder/layer_11/attention/self/key/bias/Assign4bert/encoder/layer_11/attention/self/key/bias/read:02Abert/encoder/layer_11/attention/self/key/bias/Initializer/zeros:08
?
3bert/encoder/layer_11/attention/self/value/kernel:08bert/encoder/layer_11/attention/self/value/kernel/Assign8bert/encoder/layer_11/attention/self/value/kernel/read:02Pbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal:08
?
1bert/encoder/layer_11/attention/self/value/bias:06bert/encoder/layer_11/attention/self/value/bias/Assign6bert/encoder/layer_11/attention/self/value/bias/read:02Cbert/encoder/layer_11/attention/self/value/bias/Initializer/zeros:08
?
5bert/encoder/layer_11/attention/output/dense/kernel:0:bert/encoder/layer_11/attention/output/dense/kernel/Assign:bert/encoder/layer_11/attention/output/dense/kernel/read:02Rbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal:08
?
3bert/encoder/layer_11/attention/output/dense/bias:08bert/encoder/layer_11/attention/output/dense/bias/Assign8bert/encoder/layer_11/attention/output/dense/bias/read:02Ebert/encoder/layer_11/attention/output/dense/bias/Initializer/zeros:08
?
7bert/encoder/layer_11/attention/output/LayerNorm/beta:0<bert/encoder/layer_11/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_11/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zeros:08
?
8bert/encoder/layer_11/attention/output/LayerNorm/gamma:0=bert/encoder/layer_11/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_11/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/ones:08
?
1bert/encoder/layer_11/intermediate/dense/kernel:06bert/encoder/layer_11/intermediate/dense/kernel/Assign6bert/encoder/layer_11/intermediate/dense/kernel/read:02Nbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal:08
?
/bert/encoder/layer_11/intermediate/dense/bias:04bert/encoder/layer_11/intermediate/dense/bias/Assign4bert/encoder/layer_11/intermediate/dense/bias/read:02Abert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros:08
?
+bert/encoder/layer_11/output/dense/kernel:00bert/encoder/layer_11/output/dense/kernel/Assign0bert/encoder/layer_11/output/dense/kernel/read:02Hbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal:08
?
)bert/encoder/layer_11/output/dense/bias:0.bert/encoder/layer_11/output/dense/bias/Assign.bert/encoder/layer_11/output/dense/bias/read:02;bert/encoder/layer_11/output/dense/bias/Initializer/zeros:08
?
-bert/encoder/layer_11/output/LayerNorm/beta:02bert/encoder/layer_11/output/LayerNorm/beta/Assign2bert/encoder/layer_11/output/LayerNorm/beta/read:02?bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zeros:08
?
.bert/encoder/layer_11/output/LayerNorm/gamma:03bert/encoder/layer_11/output/LayerNorm/gamma/Assign3bert/encoder/layer_11/output/LayerNorm/gamma/read:02?bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/ones:08
?
bert/pooler/dense/kernel:0bert/pooler/dense/kernel/Assignbert/pooler/dense/kernel/read:027bert/pooler/dense/kernel/Initializer/truncated_normal:08
?
bert/pooler/dense/bias:0bert/pooler/dense/bias/Assignbert/pooler/dense/bias/read:02*bert/pooler/dense/bias/Initializer/zeros:08
q
output_weights:0output_weights/Assignoutput_weights/read:02-output_weights/Initializer/truncated_normal:08
Z
output_bias:0output_bias/Assignoutput_bias/read:02output_bias/Initializer/zeros:08"??
	variables????
?
!bert/embeddings/word_embeddings:0&bert/embeddings/word_embeddings/Assign&bert/embeddings/word_embeddings/read:02>bert/embeddings/word_embeddings/Initializer/truncated_normal:08
?
'bert/embeddings/token_type_embeddings:0,bert/embeddings/token_type_embeddings/Assign,bert/embeddings/token_type_embeddings/read:02Dbert/embeddings/token_type_embeddings/Initializer/truncated_normal:08
?
%bert/embeddings/position_embeddings:0*bert/embeddings/position_embeddings/Assign*bert/embeddings/position_embeddings/read:02Bbert/embeddings/position_embeddings/Initializer/truncated_normal:08
?
 bert/embeddings/LayerNorm/beta:0%bert/embeddings/LayerNorm/beta/Assign%bert/embeddings/LayerNorm/beta/read:022bert/embeddings/LayerNorm/beta/Initializer/zeros:08
?
!bert/embeddings/LayerNorm/gamma:0&bert/embeddings/LayerNorm/gamma/Assign&bert/embeddings/LayerNorm/gamma/read:022bert/embeddings/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_0/attention/self/query/kernel:07bert/encoder/layer_0/attention/self/query/kernel/Assign7bert/encoder/layer_0/attention/self/query/kernel/read:02Obert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_0/attention/self/query/bias:05bert/encoder/layer_0/attention/self/query/bias/Assign5bert/encoder/layer_0/attention/self/query/bias/read:02Bbert/encoder/layer_0/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_0/attention/self/key/kernel:05bert/encoder/layer_0/attention/self/key/kernel/Assign5bert/encoder/layer_0/attention/self/key/kernel/read:02Mbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_0/attention/self/key/bias:03bert/encoder/layer_0/attention/self/key/bias/Assign3bert/encoder/layer_0/attention/self/key/bias/read:02@bert/encoder/layer_0/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_0/attention/self/value/kernel:07bert/encoder/layer_0/attention/self/value/kernel/Assign7bert/encoder/layer_0/attention/self/value/kernel/read:02Obert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_0/attention/self/value/bias:05bert/encoder/layer_0/attention/self/value/bias/Assign5bert/encoder/layer_0/attention/self/value/bias/read:02Bbert/encoder/layer_0/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_0/attention/output/dense/kernel:09bert/encoder/layer_0/attention/output/dense/kernel/Assign9bert/encoder/layer_0/attention/output/dense/kernel/read:02Qbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_0/attention/output/dense/bias:07bert/encoder/layer_0/attention/output/dense/bias/Assign7bert/encoder/layer_0/attention/output/dense/bias/read:02Dbert/encoder/layer_0/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_0/attention/output/LayerNorm/beta:0;bert/encoder/layer_0/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_0/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_0/attention/output/LayerNorm/gamma:0<bert/encoder/layer_0/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_0/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_0/intermediate/dense/kernel:05bert/encoder/layer_0/intermediate/dense/kernel/Assign5bert/encoder/layer_0/intermediate/dense/kernel/read:02Mbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_0/intermediate/dense/bias:03bert/encoder/layer_0/intermediate/dense/bias/Assign3bert/encoder/layer_0/intermediate/dense/bias/read:02@bert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_0/output/dense/kernel:0/bert/encoder/layer_0/output/dense/kernel/Assign/bert/encoder/layer_0/output/dense/kernel/read:02Gbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_0/output/dense/bias:0-bert/encoder/layer_0/output/dense/bias/Assign-bert/encoder/layer_0/output/dense/bias/read:02:bert/encoder/layer_0/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_0/output/LayerNorm/beta:01bert/encoder/layer_0/output/LayerNorm/beta/Assign1bert/encoder/layer_0/output/LayerNorm/beta/read:02>bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_0/output/LayerNorm/gamma:02bert/encoder/layer_0/output/LayerNorm/gamma/Assign2bert/encoder/layer_0/output/LayerNorm/gamma/read:02>bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_1/attention/self/query/kernel:07bert/encoder/layer_1/attention/self/query/kernel/Assign7bert/encoder/layer_1/attention/self/query/kernel/read:02Obert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_1/attention/self/query/bias:05bert/encoder/layer_1/attention/self/query/bias/Assign5bert/encoder/layer_1/attention/self/query/bias/read:02Bbert/encoder/layer_1/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_1/attention/self/key/kernel:05bert/encoder/layer_1/attention/self/key/kernel/Assign5bert/encoder/layer_1/attention/self/key/kernel/read:02Mbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_1/attention/self/key/bias:03bert/encoder/layer_1/attention/self/key/bias/Assign3bert/encoder/layer_1/attention/self/key/bias/read:02@bert/encoder/layer_1/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_1/attention/self/value/kernel:07bert/encoder/layer_1/attention/self/value/kernel/Assign7bert/encoder/layer_1/attention/self/value/kernel/read:02Obert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_1/attention/self/value/bias:05bert/encoder/layer_1/attention/self/value/bias/Assign5bert/encoder/layer_1/attention/self/value/bias/read:02Bbert/encoder/layer_1/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_1/attention/output/dense/kernel:09bert/encoder/layer_1/attention/output/dense/kernel/Assign9bert/encoder/layer_1/attention/output/dense/kernel/read:02Qbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_1/attention/output/dense/bias:07bert/encoder/layer_1/attention/output/dense/bias/Assign7bert/encoder/layer_1/attention/output/dense/bias/read:02Dbert/encoder/layer_1/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_1/attention/output/LayerNorm/beta:0;bert/encoder/layer_1/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_1/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_1/attention/output/LayerNorm/gamma:0<bert/encoder/layer_1/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_1/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_1/intermediate/dense/kernel:05bert/encoder/layer_1/intermediate/dense/kernel/Assign5bert/encoder/layer_1/intermediate/dense/kernel/read:02Mbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_1/intermediate/dense/bias:03bert/encoder/layer_1/intermediate/dense/bias/Assign3bert/encoder/layer_1/intermediate/dense/bias/read:02@bert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_1/output/dense/kernel:0/bert/encoder/layer_1/output/dense/kernel/Assign/bert/encoder/layer_1/output/dense/kernel/read:02Gbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_1/output/dense/bias:0-bert/encoder/layer_1/output/dense/bias/Assign-bert/encoder/layer_1/output/dense/bias/read:02:bert/encoder/layer_1/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_1/output/LayerNorm/beta:01bert/encoder/layer_1/output/LayerNorm/beta/Assign1bert/encoder/layer_1/output/LayerNorm/beta/read:02>bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_1/output/LayerNorm/gamma:02bert/encoder/layer_1/output/LayerNorm/gamma/Assign2bert/encoder/layer_1/output/LayerNorm/gamma/read:02>bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_2/attention/self/query/kernel:07bert/encoder/layer_2/attention/self/query/kernel/Assign7bert/encoder/layer_2/attention/self/query/kernel/read:02Obert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_2/attention/self/query/bias:05bert/encoder/layer_2/attention/self/query/bias/Assign5bert/encoder/layer_2/attention/self/query/bias/read:02Bbert/encoder/layer_2/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_2/attention/self/key/kernel:05bert/encoder/layer_2/attention/self/key/kernel/Assign5bert/encoder/layer_2/attention/self/key/kernel/read:02Mbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_2/attention/self/key/bias:03bert/encoder/layer_2/attention/self/key/bias/Assign3bert/encoder/layer_2/attention/self/key/bias/read:02@bert/encoder/layer_2/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_2/attention/self/value/kernel:07bert/encoder/layer_2/attention/self/value/kernel/Assign7bert/encoder/layer_2/attention/self/value/kernel/read:02Obert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_2/attention/self/value/bias:05bert/encoder/layer_2/attention/self/value/bias/Assign5bert/encoder/layer_2/attention/self/value/bias/read:02Bbert/encoder/layer_2/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_2/attention/output/dense/kernel:09bert/encoder/layer_2/attention/output/dense/kernel/Assign9bert/encoder/layer_2/attention/output/dense/kernel/read:02Qbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_2/attention/output/dense/bias:07bert/encoder/layer_2/attention/output/dense/bias/Assign7bert/encoder/layer_2/attention/output/dense/bias/read:02Dbert/encoder/layer_2/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_2/attention/output/LayerNorm/beta:0;bert/encoder/layer_2/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_2/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_2/attention/output/LayerNorm/gamma:0<bert/encoder/layer_2/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_2/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_2/intermediate/dense/kernel:05bert/encoder/layer_2/intermediate/dense/kernel/Assign5bert/encoder/layer_2/intermediate/dense/kernel/read:02Mbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_2/intermediate/dense/bias:03bert/encoder/layer_2/intermediate/dense/bias/Assign3bert/encoder/layer_2/intermediate/dense/bias/read:02@bert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_2/output/dense/kernel:0/bert/encoder/layer_2/output/dense/kernel/Assign/bert/encoder/layer_2/output/dense/kernel/read:02Gbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_2/output/dense/bias:0-bert/encoder/layer_2/output/dense/bias/Assign-bert/encoder/layer_2/output/dense/bias/read:02:bert/encoder/layer_2/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_2/output/LayerNorm/beta:01bert/encoder/layer_2/output/LayerNorm/beta/Assign1bert/encoder/layer_2/output/LayerNorm/beta/read:02>bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_2/output/LayerNorm/gamma:02bert/encoder/layer_2/output/LayerNorm/gamma/Assign2bert/encoder/layer_2/output/LayerNorm/gamma/read:02>bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_3/attention/self/query/kernel:07bert/encoder/layer_3/attention/self/query/kernel/Assign7bert/encoder/layer_3/attention/self/query/kernel/read:02Obert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_3/attention/self/query/bias:05bert/encoder/layer_3/attention/self/query/bias/Assign5bert/encoder/layer_3/attention/self/query/bias/read:02Bbert/encoder/layer_3/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_3/attention/self/key/kernel:05bert/encoder/layer_3/attention/self/key/kernel/Assign5bert/encoder/layer_3/attention/self/key/kernel/read:02Mbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_3/attention/self/key/bias:03bert/encoder/layer_3/attention/self/key/bias/Assign3bert/encoder/layer_3/attention/self/key/bias/read:02@bert/encoder/layer_3/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_3/attention/self/value/kernel:07bert/encoder/layer_3/attention/self/value/kernel/Assign7bert/encoder/layer_3/attention/self/value/kernel/read:02Obert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_3/attention/self/value/bias:05bert/encoder/layer_3/attention/self/value/bias/Assign5bert/encoder/layer_3/attention/self/value/bias/read:02Bbert/encoder/layer_3/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_3/attention/output/dense/kernel:09bert/encoder/layer_3/attention/output/dense/kernel/Assign9bert/encoder/layer_3/attention/output/dense/kernel/read:02Qbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_3/attention/output/dense/bias:07bert/encoder/layer_3/attention/output/dense/bias/Assign7bert/encoder/layer_3/attention/output/dense/bias/read:02Dbert/encoder/layer_3/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_3/attention/output/LayerNorm/beta:0;bert/encoder/layer_3/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_3/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_3/attention/output/LayerNorm/gamma:0<bert/encoder/layer_3/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_3/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_3/intermediate/dense/kernel:05bert/encoder/layer_3/intermediate/dense/kernel/Assign5bert/encoder/layer_3/intermediate/dense/kernel/read:02Mbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_3/intermediate/dense/bias:03bert/encoder/layer_3/intermediate/dense/bias/Assign3bert/encoder/layer_3/intermediate/dense/bias/read:02@bert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_3/output/dense/kernel:0/bert/encoder/layer_3/output/dense/kernel/Assign/bert/encoder/layer_3/output/dense/kernel/read:02Gbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_3/output/dense/bias:0-bert/encoder/layer_3/output/dense/bias/Assign-bert/encoder/layer_3/output/dense/bias/read:02:bert/encoder/layer_3/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_3/output/LayerNorm/beta:01bert/encoder/layer_3/output/LayerNorm/beta/Assign1bert/encoder/layer_3/output/LayerNorm/beta/read:02>bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_3/output/LayerNorm/gamma:02bert/encoder/layer_3/output/LayerNorm/gamma/Assign2bert/encoder/layer_3/output/LayerNorm/gamma/read:02>bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_4/attention/self/query/kernel:07bert/encoder/layer_4/attention/self/query/kernel/Assign7bert/encoder/layer_4/attention/self/query/kernel/read:02Obert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_4/attention/self/query/bias:05bert/encoder/layer_4/attention/self/query/bias/Assign5bert/encoder/layer_4/attention/self/query/bias/read:02Bbert/encoder/layer_4/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_4/attention/self/key/kernel:05bert/encoder/layer_4/attention/self/key/kernel/Assign5bert/encoder/layer_4/attention/self/key/kernel/read:02Mbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_4/attention/self/key/bias:03bert/encoder/layer_4/attention/self/key/bias/Assign3bert/encoder/layer_4/attention/self/key/bias/read:02@bert/encoder/layer_4/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_4/attention/self/value/kernel:07bert/encoder/layer_4/attention/self/value/kernel/Assign7bert/encoder/layer_4/attention/self/value/kernel/read:02Obert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_4/attention/self/value/bias:05bert/encoder/layer_4/attention/self/value/bias/Assign5bert/encoder/layer_4/attention/self/value/bias/read:02Bbert/encoder/layer_4/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_4/attention/output/dense/kernel:09bert/encoder/layer_4/attention/output/dense/kernel/Assign9bert/encoder/layer_4/attention/output/dense/kernel/read:02Qbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_4/attention/output/dense/bias:07bert/encoder/layer_4/attention/output/dense/bias/Assign7bert/encoder/layer_4/attention/output/dense/bias/read:02Dbert/encoder/layer_4/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_4/attention/output/LayerNorm/beta:0;bert/encoder/layer_4/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_4/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_4/attention/output/LayerNorm/gamma:0<bert/encoder/layer_4/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_4/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_4/intermediate/dense/kernel:05bert/encoder/layer_4/intermediate/dense/kernel/Assign5bert/encoder/layer_4/intermediate/dense/kernel/read:02Mbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_4/intermediate/dense/bias:03bert/encoder/layer_4/intermediate/dense/bias/Assign3bert/encoder/layer_4/intermediate/dense/bias/read:02@bert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_4/output/dense/kernel:0/bert/encoder/layer_4/output/dense/kernel/Assign/bert/encoder/layer_4/output/dense/kernel/read:02Gbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_4/output/dense/bias:0-bert/encoder/layer_4/output/dense/bias/Assign-bert/encoder/layer_4/output/dense/bias/read:02:bert/encoder/layer_4/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_4/output/LayerNorm/beta:01bert/encoder/layer_4/output/LayerNorm/beta/Assign1bert/encoder/layer_4/output/LayerNorm/beta/read:02>bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_4/output/LayerNorm/gamma:02bert/encoder/layer_4/output/LayerNorm/gamma/Assign2bert/encoder/layer_4/output/LayerNorm/gamma/read:02>bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_5/attention/self/query/kernel:07bert/encoder/layer_5/attention/self/query/kernel/Assign7bert/encoder/layer_5/attention/self/query/kernel/read:02Obert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_5/attention/self/query/bias:05bert/encoder/layer_5/attention/self/query/bias/Assign5bert/encoder/layer_5/attention/self/query/bias/read:02Bbert/encoder/layer_5/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_5/attention/self/key/kernel:05bert/encoder/layer_5/attention/self/key/kernel/Assign5bert/encoder/layer_5/attention/self/key/kernel/read:02Mbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_5/attention/self/key/bias:03bert/encoder/layer_5/attention/self/key/bias/Assign3bert/encoder/layer_5/attention/self/key/bias/read:02@bert/encoder/layer_5/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_5/attention/self/value/kernel:07bert/encoder/layer_5/attention/self/value/kernel/Assign7bert/encoder/layer_5/attention/self/value/kernel/read:02Obert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_5/attention/self/value/bias:05bert/encoder/layer_5/attention/self/value/bias/Assign5bert/encoder/layer_5/attention/self/value/bias/read:02Bbert/encoder/layer_5/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_5/attention/output/dense/kernel:09bert/encoder/layer_5/attention/output/dense/kernel/Assign9bert/encoder/layer_5/attention/output/dense/kernel/read:02Qbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_5/attention/output/dense/bias:07bert/encoder/layer_5/attention/output/dense/bias/Assign7bert/encoder/layer_5/attention/output/dense/bias/read:02Dbert/encoder/layer_5/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_5/attention/output/LayerNorm/beta:0;bert/encoder/layer_5/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_5/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_5/attention/output/LayerNorm/gamma:0<bert/encoder/layer_5/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_5/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_5/intermediate/dense/kernel:05bert/encoder/layer_5/intermediate/dense/kernel/Assign5bert/encoder/layer_5/intermediate/dense/kernel/read:02Mbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_5/intermediate/dense/bias:03bert/encoder/layer_5/intermediate/dense/bias/Assign3bert/encoder/layer_5/intermediate/dense/bias/read:02@bert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_5/output/dense/kernel:0/bert/encoder/layer_5/output/dense/kernel/Assign/bert/encoder/layer_5/output/dense/kernel/read:02Gbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_5/output/dense/bias:0-bert/encoder/layer_5/output/dense/bias/Assign-bert/encoder/layer_5/output/dense/bias/read:02:bert/encoder/layer_5/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_5/output/LayerNorm/beta:01bert/encoder/layer_5/output/LayerNorm/beta/Assign1bert/encoder/layer_5/output/LayerNorm/beta/read:02>bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_5/output/LayerNorm/gamma:02bert/encoder/layer_5/output/LayerNorm/gamma/Assign2bert/encoder/layer_5/output/LayerNorm/gamma/read:02>bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_6/attention/self/query/kernel:07bert/encoder/layer_6/attention/self/query/kernel/Assign7bert/encoder/layer_6/attention/self/query/kernel/read:02Obert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_6/attention/self/query/bias:05bert/encoder/layer_6/attention/self/query/bias/Assign5bert/encoder/layer_6/attention/self/query/bias/read:02Bbert/encoder/layer_6/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_6/attention/self/key/kernel:05bert/encoder/layer_6/attention/self/key/kernel/Assign5bert/encoder/layer_6/attention/self/key/kernel/read:02Mbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_6/attention/self/key/bias:03bert/encoder/layer_6/attention/self/key/bias/Assign3bert/encoder/layer_6/attention/self/key/bias/read:02@bert/encoder/layer_6/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_6/attention/self/value/kernel:07bert/encoder/layer_6/attention/self/value/kernel/Assign7bert/encoder/layer_6/attention/self/value/kernel/read:02Obert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_6/attention/self/value/bias:05bert/encoder/layer_6/attention/self/value/bias/Assign5bert/encoder/layer_6/attention/self/value/bias/read:02Bbert/encoder/layer_6/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_6/attention/output/dense/kernel:09bert/encoder/layer_6/attention/output/dense/kernel/Assign9bert/encoder/layer_6/attention/output/dense/kernel/read:02Qbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_6/attention/output/dense/bias:07bert/encoder/layer_6/attention/output/dense/bias/Assign7bert/encoder/layer_6/attention/output/dense/bias/read:02Dbert/encoder/layer_6/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_6/attention/output/LayerNorm/beta:0;bert/encoder/layer_6/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_6/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_6/attention/output/LayerNorm/gamma:0<bert/encoder/layer_6/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_6/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_6/intermediate/dense/kernel:05bert/encoder/layer_6/intermediate/dense/kernel/Assign5bert/encoder/layer_6/intermediate/dense/kernel/read:02Mbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_6/intermediate/dense/bias:03bert/encoder/layer_6/intermediate/dense/bias/Assign3bert/encoder/layer_6/intermediate/dense/bias/read:02@bert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_6/output/dense/kernel:0/bert/encoder/layer_6/output/dense/kernel/Assign/bert/encoder/layer_6/output/dense/kernel/read:02Gbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_6/output/dense/bias:0-bert/encoder/layer_6/output/dense/bias/Assign-bert/encoder/layer_6/output/dense/bias/read:02:bert/encoder/layer_6/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_6/output/LayerNorm/beta:01bert/encoder/layer_6/output/LayerNorm/beta/Assign1bert/encoder/layer_6/output/LayerNorm/beta/read:02>bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_6/output/LayerNorm/gamma:02bert/encoder/layer_6/output/LayerNorm/gamma/Assign2bert/encoder/layer_6/output/LayerNorm/gamma/read:02>bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_7/attention/self/query/kernel:07bert/encoder/layer_7/attention/self/query/kernel/Assign7bert/encoder/layer_7/attention/self/query/kernel/read:02Obert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_7/attention/self/query/bias:05bert/encoder/layer_7/attention/self/query/bias/Assign5bert/encoder/layer_7/attention/self/query/bias/read:02Bbert/encoder/layer_7/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_7/attention/self/key/kernel:05bert/encoder/layer_7/attention/self/key/kernel/Assign5bert/encoder/layer_7/attention/self/key/kernel/read:02Mbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_7/attention/self/key/bias:03bert/encoder/layer_7/attention/self/key/bias/Assign3bert/encoder/layer_7/attention/self/key/bias/read:02@bert/encoder/layer_7/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_7/attention/self/value/kernel:07bert/encoder/layer_7/attention/self/value/kernel/Assign7bert/encoder/layer_7/attention/self/value/kernel/read:02Obert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_7/attention/self/value/bias:05bert/encoder/layer_7/attention/self/value/bias/Assign5bert/encoder/layer_7/attention/self/value/bias/read:02Bbert/encoder/layer_7/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_7/attention/output/dense/kernel:09bert/encoder/layer_7/attention/output/dense/kernel/Assign9bert/encoder/layer_7/attention/output/dense/kernel/read:02Qbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_7/attention/output/dense/bias:07bert/encoder/layer_7/attention/output/dense/bias/Assign7bert/encoder/layer_7/attention/output/dense/bias/read:02Dbert/encoder/layer_7/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_7/attention/output/LayerNorm/beta:0;bert/encoder/layer_7/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_7/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_7/attention/output/LayerNorm/gamma:0<bert/encoder/layer_7/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_7/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_7/intermediate/dense/kernel:05bert/encoder/layer_7/intermediate/dense/kernel/Assign5bert/encoder/layer_7/intermediate/dense/kernel/read:02Mbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_7/intermediate/dense/bias:03bert/encoder/layer_7/intermediate/dense/bias/Assign3bert/encoder/layer_7/intermediate/dense/bias/read:02@bert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_7/output/dense/kernel:0/bert/encoder/layer_7/output/dense/kernel/Assign/bert/encoder/layer_7/output/dense/kernel/read:02Gbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_7/output/dense/bias:0-bert/encoder/layer_7/output/dense/bias/Assign-bert/encoder/layer_7/output/dense/bias/read:02:bert/encoder/layer_7/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_7/output/LayerNorm/beta:01bert/encoder/layer_7/output/LayerNorm/beta/Assign1bert/encoder/layer_7/output/LayerNorm/beta/read:02>bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_7/output/LayerNorm/gamma:02bert/encoder/layer_7/output/LayerNorm/gamma/Assign2bert/encoder/layer_7/output/LayerNorm/gamma/read:02>bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_8/attention/self/query/kernel:07bert/encoder/layer_8/attention/self/query/kernel/Assign7bert/encoder/layer_8/attention/self/query/kernel/read:02Obert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_8/attention/self/query/bias:05bert/encoder/layer_8/attention/self/query/bias/Assign5bert/encoder/layer_8/attention/self/query/bias/read:02Bbert/encoder/layer_8/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_8/attention/self/key/kernel:05bert/encoder/layer_8/attention/self/key/kernel/Assign5bert/encoder/layer_8/attention/self/key/kernel/read:02Mbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_8/attention/self/key/bias:03bert/encoder/layer_8/attention/self/key/bias/Assign3bert/encoder/layer_8/attention/self/key/bias/read:02@bert/encoder/layer_8/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_8/attention/self/value/kernel:07bert/encoder/layer_8/attention/self/value/kernel/Assign7bert/encoder/layer_8/attention/self/value/kernel/read:02Obert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_8/attention/self/value/bias:05bert/encoder/layer_8/attention/self/value/bias/Assign5bert/encoder/layer_8/attention/self/value/bias/read:02Bbert/encoder/layer_8/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_8/attention/output/dense/kernel:09bert/encoder/layer_8/attention/output/dense/kernel/Assign9bert/encoder/layer_8/attention/output/dense/kernel/read:02Qbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_8/attention/output/dense/bias:07bert/encoder/layer_8/attention/output/dense/bias/Assign7bert/encoder/layer_8/attention/output/dense/bias/read:02Dbert/encoder/layer_8/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_8/attention/output/LayerNorm/beta:0;bert/encoder/layer_8/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_8/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_8/attention/output/LayerNorm/gamma:0<bert/encoder/layer_8/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_8/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_8/intermediate/dense/kernel:05bert/encoder/layer_8/intermediate/dense/kernel/Assign5bert/encoder/layer_8/intermediate/dense/kernel/read:02Mbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_8/intermediate/dense/bias:03bert/encoder/layer_8/intermediate/dense/bias/Assign3bert/encoder/layer_8/intermediate/dense/bias/read:02@bert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_8/output/dense/kernel:0/bert/encoder/layer_8/output/dense/kernel/Assign/bert/encoder/layer_8/output/dense/kernel/read:02Gbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_8/output/dense/bias:0-bert/encoder/layer_8/output/dense/bias/Assign-bert/encoder/layer_8/output/dense/bias/read:02:bert/encoder/layer_8/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_8/output/LayerNorm/beta:01bert/encoder/layer_8/output/LayerNorm/beta/Assign1bert/encoder/layer_8/output/LayerNorm/beta/read:02>bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_8/output/LayerNorm/gamma:02bert/encoder/layer_8/output/LayerNorm/gamma/Assign2bert/encoder/layer_8/output/LayerNorm/gamma/read:02>bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/ones:08
?
2bert/encoder/layer_9/attention/self/query/kernel:07bert/encoder/layer_9/attention/self/query/kernel/Assign7bert/encoder/layer_9/attention/self/query/kernel/read:02Obert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_9/attention/self/query/bias:05bert/encoder/layer_9/attention/self/query/bias/Assign5bert/encoder/layer_9/attention/self/query/bias/read:02Bbert/encoder/layer_9/attention/self/query/bias/Initializer/zeros:08
?
0bert/encoder/layer_9/attention/self/key/kernel:05bert/encoder/layer_9/attention/self/key/kernel/Assign5bert/encoder/layer_9/attention/self/key/kernel/read:02Mbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_9/attention/self/key/bias:03bert/encoder/layer_9/attention/self/key/bias/Assign3bert/encoder/layer_9/attention/self/key/bias/read:02@bert/encoder/layer_9/attention/self/key/bias/Initializer/zeros:08
?
2bert/encoder/layer_9/attention/self/value/kernel:07bert/encoder/layer_9/attention/self/value/kernel/Assign7bert/encoder/layer_9/attention/self/value/kernel/read:02Obert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal:08
?
0bert/encoder/layer_9/attention/self/value/bias:05bert/encoder/layer_9/attention/self/value/bias/Assign5bert/encoder/layer_9/attention/self/value/bias/read:02Bbert/encoder/layer_9/attention/self/value/bias/Initializer/zeros:08
?
4bert/encoder/layer_9/attention/output/dense/kernel:09bert/encoder/layer_9/attention/output/dense/kernel/Assign9bert/encoder/layer_9/attention/output/dense/kernel/read:02Qbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal:08
?
2bert/encoder/layer_9/attention/output/dense/bias:07bert/encoder/layer_9/attention/output/dense/bias/Assign7bert/encoder/layer_9/attention/output/dense/bias/read:02Dbert/encoder/layer_9/attention/output/dense/bias/Initializer/zeros:08
?
6bert/encoder/layer_9/attention/output/LayerNorm/beta:0;bert/encoder/layer_9/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_9/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zeros:08
?
7bert/encoder/layer_9/attention/output/LayerNorm/gamma:0<bert/encoder/layer_9/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_9/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/ones:08
?
0bert/encoder/layer_9/intermediate/dense/kernel:05bert/encoder/layer_9/intermediate/dense/kernel/Assign5bert/encoder/layer_9/intermediate/dense/kernel/read:02Mbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal:08
?
.bert/encoder/layer_9/intermediate/dense/bias:03bert/encoder/layer_9/intermediate/dense/bias/Assign3bert/encoder/layer_9/intermediate/dense/bias/read:02@bert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros:08
?
*bert/encoder/layer_9/output/dense/kernel:0/bert/encoder/layer_9/output/dense/kernel/Assign/bert/encoder/layer_9/output/dense/kernel/read:02Gbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal:08
?
(bert/encoder/layer_9/output/dense/bias:0-bert/encoder/layer_9/output/dense/bias/Assign-bert/encoder/layer_9/output/dense/bias/read:02:bert/encoder/layer_9/output/dense/bias/Initializer/zeros:08
?
,bert/encoder/layer_9/output/LayerNorm/beta:01bert/encoder/layer_9/output/LayerNorm/beta/Assign1bert/encoder/layer_9/output/LayerNorm/beta/read:02>bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zeros:08
?
-bert/encoder/layer_9/output/LayerNorm/gamma:02bert/encoder/layer_9/output/LayerNorm/gamma/Assign2bert/encoder/layer_9/output/LayerNorm/gamma/read:02>bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/ones:08
?
3bert/encoder/layer_10/attention/self/query/kernel:08bert/encoder/layer_10/attention/self/query/kernel/Assign8bert/encoder/layer_10/attention/self/query/kernel/read:02Pbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal:08
?
1bert/encoder/layer_10/attention/self/query/bias:06bert/encoder/layer_10/attention/self/query/bias/Assign6bert/encoder/layer_10/attention/self/query/bias/read:02Cbert/encoder/layer_10/attention/self/query/bias/Initializer/zeros:08
?
1bert/encoder/layer_10/attention/self/key/kernel:06bert/encoder/layer_10/attention/self/key/kernel/Assign6bert/encoder/layer_10/attention/self/key/kernel/read:02Nbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal:08
?
/bert/encoder/layer_10/attention/self/key/bias:04bert/encoder/layer_10/attention/self/key/bias/Assign4bert/encoder/layer_10/attention/self/key/bias/read:02Abert/encoder/layer_10/attention/self/key/bias/Initializer/zeros:08
?
3bert/encoder/layer_10/attention/self/value/kernel:08bert/encoder/layer_10/attention/self/value/kernel/Assign8bert/encoder/layer_10/attention/self/value/kernel/read:02Pbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal:08
?
1bert/encoder/layer_10/attention/self/value/bias:06bert/encoder/layer_10/attention/self/value/bias/Assign6bert/encoder/layer_10/attention/self/value/bias/read:02Cbert/encoder/layer_10/attention/self/value/bias/Initializer/zeros:08
?
5bert/encoder/layer_10/attention/output/dense/kernel:0:bert/encoder/layer_10/attention/output/dense/kernel/Assign:bert/encoder/layer_10/attention/output/dense/kernel/read:02Rbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal:08
?
3bert/encoder/layer_10/attention/output/dense/bias:08bert/encoder/layer_10/attention/output/dense/bias/Assign8bert/encoder/layer_10/attention/output/dense/bias/read:02Ebert/encoder/layer_10/attention/output/dense/bias/Initializer/zeros:08
?
7bert/encoder/layer_10/attention/output/LayerNorm/beta:0<bert/encoder/layer_10/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_10/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zeros:08
?
8bert/encoder/layer_10/attention/output/LayerNorm/gamma:0=bert/encoder/layer_10/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_10/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/ones:08
?
1bert/encoder/layer_10/intermediate/dense/kernel:06bert/encoder/layer_10/intermediate/dense/kernel/Assign6bert/encoder/layer_10/intermediate/dense/kernel/read:02Nbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal:08
?
/bert/encoder/layer_10/intermediate/dense/bias:04bert/encoder/layer_10/intermediate/dense/bias/Assign4bert/encoder/layer_10/intermediate/dense/bias/read:02Abert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros:08
?
+bert/encoder/layer_10/output/dense/kernel:00bert/encoder/layer_10/output/dense/kernel/Assign0bert/encoder/layer_10/output/dense/kernel/read:02Hbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal:08
?
)bert/encoder/layer_10/output/dense/bias:0.bert/encoder/layer_10/output/dense/bias/Assign.bert/encoder/layer_10/output/dense/bias/read:02;bert/encoder/layer_10/output/dense/bias/Initializer/zeros:08
?
-bert/encoder/layer_10/output/LayerNorm/beta:02bert/encoder/layer_10/output/LayerNorm/beta/Assign2bert/encoder/layer_10/output/LayerNorm/beta/read:02?bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zeros:08
?
.bert/encoder/layer_10/output/LayerNorm/gamma:03bert/encoder/layer_10/output/LayerNorm/gamma/Assign3bert/encoder/layer_10/output/LayerNorm/gamma/read:02?bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/ones:08
?
3bert/encoder/layer_11/attention/self/query/kernel:08bert/encoder/layer_11/attention/self/query/kernel/Assign8bert/encoder/layer_11/attention/self/query/kernel/read:02Pbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal:08
?
1bert/encoder/layer_11/attention/self/query/bias:06bert/encoder/layer_11/attention/self/query/bias/Assign6bert/encoder/layer_11/attention/self/query/bias/read:02Cbert/encoder/layer_11/attention/self/query/bias/Initializer/zeros:08
?
1bert/encoder/layer_11/attention/self/key/kernel:06bert/encoder/layer_11/attention/self/key/kernel/Assign6bert/encoder/layer_11/attention/self/key/kernel/read:02Nbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal:08
?
/bert/encoder/layer_11/attention/self/key/bias:04bert/encoder/layer_11/attention/self/key/bias/Assign4bert/encoder/layer_11/attention/self/key/bias/read:02Abert/encoder/layer_11/attention/self/key/bias/Initializer/zeros:08
?
3bert/encoder/layer_11/attention/self/value/kernel:08bert/encoder/layer_11/attention/self/value/kernel/Assign8bert/encoder/layer_11/attention/self/value/kernel/read:02Pbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal:08
?
1bert/encoder/layer_11/attention/self/value/bias:06bert/encoder/layer_11/attention/self/value/bias/Assign6bert/encoder/layer_11/attention/self/value/bias/read:02Cbert/encoder/layer_11/attention/self/value/bias/Initializer/zeros:08
?
5bert/encoder/layer_11/attention/output/dense/kernel:0:bert/encoder/layer_11/attention/output/dense/kernel/Assign:bert/encoder/layer_11/attention/output/dense/kernel/read:02Rbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal:08
?
3bert/encoder/layer_11/attention/output/dense/bias:08bert/encoder/layer_11/attention/output/dense/bias/Assign8bert/encoder/layer_11/attention/output/dense/bias/read:02Ebert/encoder/layer_11/attention/output/dense/bias/Initializer/zeros:08
?
7bert/encoder/layer_11/attention/output/LayerNorm/beta:0<bert/encoder/layer_11/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_11/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zeros:08
?
8bert/encoder/layer_11/attention/output/LayerNorm/gamma:0=bert/encoder/layer_11/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_11/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/ones:08
?
1bert/encoder/layer_11/intermediate/dense/kernel:06bert/encoder/layer_11/intermediate/dense/kernel/Assign6bert/encoder/layer_11/intermediate/dense/kernel/read:02Nbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal:08
?
/bert/encoder/layer_11/intermediate/dense/bias:04bert/encoder/layer_11/intermediate/dense/bias/Assign4bert/encoder/layer_11/intermediate/dense/bias/read:02Abert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros:08
?
+bert/encoder/layer_11/output/dense/kernel:00bert/encoder/layer_11/output/dense/kernel/Assign0bert/encoder/layer_11/output/dense/kernel/read:02Hbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal:08
?
)bert/encoder/layer_11/output/dense/bias:0.bert/encoder/layer_11/output/dense/bias/Assign.bert/encoder/layer_11/output/dense/bias/read:02;bert/encoder/layer_11/output/dense/bias/Initializer/zeros:08
?
-bert/encoder/layer_11/output/LayerNorm/beta:02bert/encoder/layer_11/output/LayerNorm/beta/Assign2bert/encoder/layer_11/output/LayerNorm/beta/read:02?bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zeros:08
?
.bert/encoder/layer_11/output/LayerNorm/gamma:03bert/encoder/layer_11/output/LayerNorm/gamma/Assign3bert/encoder/layer_11/output/LayerNorm/gamma/read:02?bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/ones:08
?
bert/pooler/dense/kernel:0bert/pooler/dense/kernel/Assignbert/pooler/dense/kernel/read:027bert/pooler/dense/kernel/Initializer/truncated_normal:08
?
bert/pooler/dense/bias:0bert/pooler/dense/bias/Assignbert/pooler/dense/bias/read:02*bert/pooler/dense/bias/Initializer/zeros:08
q
output_weights:0output_weights/Assignoutput_weights/read:02-output_weights/Initializer/truncated_normal:08
Z
output_bias:0output_bias/Assignoutput_bias/read:02output_bias/Initializer/zeros:08*?
serving_default?
+
segment_ids
segment_ids:0		?

vals
vals:0
)

input_mask
input_mask:0		?
'
	input_ids
input_ids:0		?%
	pred_vals
loss/Squeeze:0tensorflow/serving/predict