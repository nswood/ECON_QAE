¡
Í
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

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
$
DisableCopyOnRead
resource
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
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
2
Round
x"T
y"T"
Ttype:
2
	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
÷
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628ì
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
x
serving_default_input_1Placeholder*&
_output_shapes
:	*
dtype0*
shape:	
è
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasdense/kernel
dense/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_1414

NoOpNoOp
É+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*+
valueú*B÷* Bð*

layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
	layer-8

layer-9
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 

	keras_api* 
­
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

activation
	quantizer* 
¾
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#kernel_quantizer
$bias_quantizer
#kernel_quantizer_internal
$bias_quantizer_internal
%
quantizers

&kernel
'bias
 (_jit_compiled_convolution_op*
­
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/
activation
/	quantizer* 

0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 

6	keras_api* 

7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=kernel_quantizer
>bias_quantizer
=kernel_quantizer_internal
>bias_quantizer_internal
?
quantizers

@kernel
Abias*

B	keras_api* 
­
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I
activation
I	quantizer* 

J	keras_api* 

K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
 
&0
'1
@2
A3*
 
&0
'1
@2
A3*
* 
°
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Vtrace_0
Wtrace_1* 

Xtrace_0
Ytrace_1* 
* 

Zserving_default* 
* 
* 
* 
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

`trace_0* 

atrace_0* 
* 

&0
'1*

&0
'1*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
* 
* 

#0
$1* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

ntrace_0* 

otrace_0* 
* 
* 
* 
* 

pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

utrace_0* 

vtrace_0* 
* 

@0
A1*

@0
A1*
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

|trace_0* 

}trace_0* 
* 
* 

=0
>1* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
Z
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
11*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ö
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasdense/kernel
dense/biasConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_1746
Ñ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasdense/kernel
dense/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_1767«¯
ñ
V
*__inference_concatenate_layer_call_fn_1693
inputs_0
inputs_1
identity·
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1297W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:::HD

_output_shapes

:
"
_user_specified_name
inputs_1:H D

_output_shapes

:
"
_user_specified_name
inputs_0
Á
h
L__inference_input_quantization_layer_call_and_return_conditional_losses_1450

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CS
mulMulinputsmul/y:output:0*
T0*&
_output_shapes
:V
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:Z
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:E
Neg_1Neginputs*
T0*&
_output_shapes
:U
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:`
add_3AddV2inputsStopGradient_1:output:0*
T0*&
_output_shapes
:P
IdentityIdentity	add_3:z:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
¢
i
M__inference_latent_quantization_layer_call_and_return_conditional_losses_1687

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CK
mulMulinputsmul/y:output:0*
T0*
_output_shapes

:N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ãv
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:=
Neg_1Neginputs*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:X
add_3AddV2inputsStopGradient_1:output:0*
T0*
_output_shapes

:H
IdentityIdentity	add_3:z:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs

N
2__inference_latent_quantization_layer_call_fn_1656

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_latent_quantization_layer_call_and_return_conditional_losses_1287W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs
2
£
?__inference_dense_layer_call_and_return_conditional_losses_1651

inputs*
readvariableop_resource:	'
readvariableop_3_resource:
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B\
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:	O
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes
:	A
NegNegtruediv:z:0*
T0*
_output_shapes
:	E
RoundRoundtruediv:z:0*
T0*
_output_shapes
:	J
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
:	O
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
:	\
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
:	\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øAw
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Âw
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:	S
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes
:	P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B_
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes
:	L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes
:	i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0P
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
:	N
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes
:	L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes
:	S
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes
:	i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0k
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes
:	L
MatMulMatMulinputs	add_3:z:0*
T0*
_output_shapes

:I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Âx
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:X
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:G C

_output_shapes
:	
 
_user_specified_nameinputs

o
E__inference_concatenate_layer_call_and_return_conditional_losses_1297

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :l
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*
_output_shapes

:N
IdentityIdentityconcat:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:::FB

_output_shapes

:
 
_user_specified_nameinputs:F B

_output_shapes

:
 
_user_specified_nameinputs
û3
«
@__inference_conv2d_layer_call_and_return_conditional_losses_1134

inputs1
readvariableop_resource:'
readvariableop_3_resource:
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: n
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bc
mulMulReadVariableOp:value:0mul/y:output:0*
T0*&
_output_shapes
:V
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øA~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:Z
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0W
Neg_1NegReadVariableOp_1:value:0*
T0*&
_output_shapes
:U
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_2ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*&
_output_shapes
:
convolutionConv2Dinputs	add_3:z:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Âx
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:d
BiasAddBiasAddconvolution:output:0	add_7:z:0*
T0*&
_output_shapes
:^
IdentityIdentityBiasAdd:output:0^NoOp*
T0*&
_output_shapes
:
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:N J
&
_output_shapes
:
 
_user_specified_nameinputs

>
"__inference_act_layer_call_fn_1532

inputs
identityª
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_act_layer_call_and_return_conditional_losses_1170_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
¢
i
M__inference_latent_quantization_layer_call_and_return_conditional_losses_1287

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CK
mulMulinputsmul/y:output:0*
T0*
_output_shapes

:N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ãv
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:=
Neg_1Neginputs*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:X
add_3AddV2inputsStopGradient_1:output:0*
T0*
_output_shapes

:H
IdentityIdentity	add_3:z:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::F B

_output_shapes

:
 
_user_specified_nameinputs

]
A__inference_flatten_layer_call_and_return_conditional_losses_1181

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   T
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	P
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs

B
&__inference_flatten_layer_call_fn_1568

inputs
identity§
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1181X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs

]
A__inference_flatten_layer_call_and_return_conditional_losses_1574

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   T
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	P
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
û3
«
@__inference_conv2d_layer_call_and_return_conditional_losses_1527

inputs1
readvariableop_resource:'
readvariableop_3_resource:
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: n
ReadVariableOpReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bc
mulMulReadVariableOp:value:0mul/y:output:0*
T0*&
_output_shapes
:V
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øA~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:Z
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Bf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_1ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0W
Neg_1NegReadVariableOp_1:value:0*
T0*&
_output_shapes
:U
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:p
ReadVariableOp_2ReadVariableOpreadvariableop_resource*&
_output_shapes
:*
dtype0r
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*&
_output_shapes
:
convolutionConv2Dinputs	add_3:z:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Âx
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:d
BiasAddBiasAddconvolution:output:0	add_7:z:0*
T0*&
_output_shapes
:^
IdentityIdentityBiasAdd:output:0^NoOp*
T0*&
_output_shapes
:
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:N J
&
_output_shapes
:
 
_user_specified_nameinputs

q
E__inference_concatenate_layer_call_and_return_conditional_losses_1700
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :n
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*
_output_shapes

:N
IdentityIdentityconcat:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:::HD

_output_shapes

:
"
_user_specified_name
inputs_1:H D

_output_shapes

:
"
_user_specified_name
inputs_0
2
£
?__inference_dense_layer_call_and_return_conditional_losses_1251

inputs*
readvariableop_resource:	'
readvariableop_3_resource:
identity¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢ReadVariableOp_4¢ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B : K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: g
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B\
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes
:	O
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes
:	A
NegNegtruediv:z:0*
T0*
_output_shapes
:	E
RoundRoundtruediv:z:0*
T0*
_output_shapes
:	J
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes
:	O
StopGradientStopGradientadd:z:0*
T0*
_output_shapes
:	\
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes
:	\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øAw
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Âw
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:	S
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes
:	P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B_
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes
:	L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes
:	i
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0P
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes
:	N
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes
:	L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes
:	S
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes
:	i
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0k
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes
:	L
MatMulMatMulinputs	add_3:z:0*
T0*
_output_shapes

:I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øAv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Âx
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   BZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:X
BiasAddBiasAddMatMul:product:0	add_7:z:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:G C

_output_shapes
:	
 
_user_specified_nameinputs
Òò
¸
__inference__wrapped_model_1028
input_1@
&encoder_conv2d_readvariableop_resource:6
(encoder_conv2d_readvariableop_3_resource:8
%encoder_dense_readvariableop_resource:	5
'encoder_dense_readvariableop_3_resource:
identity¢encoder/conv2d/ReadVariableOp¢encoder/conv2d/ReadVariableOp_1¢encoder/conv2d/ReadVariableOp_2¢encoder/conv2d/ReadVariableOp_3¢encoder/conv2d/ReadVariableOp_4¢encoder/conv2d/ReadVariableOp_5¢encoder/dense/ReadVariableOp¢encoder/dense/ReadVariableOp_1¢encoder/dense/ReadVariableOp_2¢encoder/dense/ReadVariableOp_3¢encoder/dense/ReadVariableOp_4¢encoder/dense/ReadVariableOp_5
4encoder/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
6encoder/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
6encoder/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
.encoder/tf.__operators__.getitem/strided_sliceStridedSliceinput_1=encoder/tf.__operators__.getitem/strided_slice/stack:output:0?encoder/tf.__operators__.getitem/strided_slice/stack_1:output:0?encoder/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_maskb
 encoder/input_quantization/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :b
 encoder/input_quantization/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
encoder/input_quantization/PowPow)encoder/input_quantization/Pow/x:output:0)encoder/input_quantization/Pow/y:output:0*
T0*
_output_shapes
: {
encoder/input_quantization/CastCast"encoder/input_quantization/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: e
 encoder/input_quantization/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cº
encoder/input_quantization/mulMul7encoder/tf.__operators__.getitem/strided_slice:output:0)encoder/input_quantization/mul/y:output:0*
T0*&
_output_shapes
:§
"encoder/input_quantization/truedivRealDiv"encoder/input_quantization/mul:z:0#encoder/input_quantization/Cast:y:0*
T0*&
_output_shapes
:~
encoder/input_quantization/NegNeg&encoder/input_quantization/truediv:z:0*
T0*&
_output_shapes
:
 encoder/input_quantization/RoundRound&encoder/input_quantization/truediv:z:0*
T0*&
_output_shapes
:¢
encoder/input_quantization/addAddV2"encoder/input_quantization/Neg:y:0$encoder/input_quantization/Round:y:0*
T0*&
_output_shapes
:
'encoder/input_quantization/StopGradientStopGradient"encoder/input_quantization/add:z:0*
T0*&
_output_shapes
:´
 encoder/input_quantization/add_1AddV2&encoder/input_quantization/truediv:z:00encoder/input_quantization/StopGradient:output:0*
T0*&
_output_shapes
:w
2encoder/input_quantization/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þBÏ
0encoder/input_quantization/clip_by_value/MinimumMinimum$encoder/input_quantization/add_1:z:0;encoder/input_quantization/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:o
*encoder/input_quantization/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ÃÏ
(encoder/input_quantization/clip_by_valueMaximum4encoder/input_quantization/clip_by_value/Minimum:z:03encoder/input_quantization/clip_by_value/y:output:0*
T0*&
_output_shapes
:«
 encoder/input_quantization/mul_1Mul#encoder/input_quantization/Cast:y:0,encoder/input_quantization/clip_by_value:z:0*
T0*&
_output_shapes
:k
&encoder/input_quantization/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C·
$encoder/input_quantization/truediv_1RealDiv$encoder/input_quantization/mul_1:z:0/encoder/input_quantization/truediv_1/y:output:0*
T0*&
_output_shapes
:g
"encoder/input_quantization/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
 encoder/input_quantization/mul_2Mul+encoder/input_quantization/mul_2/x:output:0(encoder/input_quantization/truediv_1:z:0*
T0*&
_output_shapes
:
 encoder/input_quantization/Neg_1Neg7encoder/tf.__operators__.getitem/strided_slice:output:0*
T0*&
_output_shapes
:¦
 encoder/input_quantization/add_2AddV2$encoder/input_quantization/Neg_1:y:0$encoder/input_quantization/mul_2:z:0*
T0*&
_output_shapes
:g
"encoder/input_quantization/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?«
 encoder/input_quantization/mul_3Mul+encoder/input_quantization/mul_3/x:output:0$encoder/input_quantization/add_2:z:0*
T0*&
_output_shapes
:
)encoder/input_quantization/StopGradient_1StopGradient$encoder/input_quantization/mul_3:z:0*
T0*&
_output_shapes
:Ç
 encoder/input_quantization/add_3AddV27encoder/tf.__operators__.getitem/strided_slice:output:02encoder/input_quantization/StopGradient_1:output:0*
T0*&
_output_shapes
:V
encoder/conv2d/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :V
encoder/conv2d/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : x
encoder/conv2d/PowPowencoder/conv2d/Pow/x:output:0encoder/conv2d/Pow/y:output:0*
T0*
_output_shapes
: c
encoder/conv2d/CastCastencoder/conv2d/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: 
encoder/conv2d/ReadVariableOpReadVariableOp&encoder_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Y
encoder/conv2d/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
encoder/conv2d/mulMul%encoder/conv2d/ReadVariableOp:value:0encoder/conv2d/mul/y:output:0*
T0*&
_output_shapes
:
encoder/conv2d/truedivRealDivencoder/conv2d/mul:z:0encoder/conv2d/Cast:y:0*
T0*&
_output_shapes
:f
encoder/conv2d/NegNegencoder/conv2d/truediv:z:0*
T0*&
_output_shapes
:j
encoder/conv2d/RoundRoundencoder/conv2d/truediv:z:0*
T0*&
_output_shapes
:~
encoder/conv2d/addAddV2encoder/conv2d/Neg:y:0encoder/conv2d/Round:y:0*
T0*&
_output_shapes
:t
encoder/conv2d/StopGradientStopGradientencoder/conv2d/add:z:0*
T0*&
_output_shapes
:
encoder/conv2d/add_1AddV2encoder/conv2d/truediv:z:0$encoder/conv2d/StopGradient:output:0*
T0*&
_output_shapes
:k
&encoder/conv2d/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øA«
$encoder/conv2d/clip_by_value/MinimumMinimumencoder/conv2d/add_1:z:0/encoder/conv2d/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:c
encoder/conv2d/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â«
encoder/conv2d/clip_by_valueMaximum(encoder/conv2d/clip_by_value/Minimum:z:0'encoder/conv2d/clip_by_value/y:output:0*
T0*&
_output_shapes
:
encoder/conv2d/mul_1Mulencoder/conv2d/Cast:y:0 encoder/conv2d/clip_by_value:z:0*
T0*&
_output_shapes
:_
encoder/conv2d/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
encoder/conv2d/truediv_1RealDivencoder/conv2d/mul_1:z:0#encoder/conv2d/truediv_1/y:output:0*
T0*&
_output_shapes
:[
encoder/conv2d/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
encoder/conv2d/mul_2Mulencoder/conv2d/mul_2/x:output:0encoder/conv2d/truediv_1:z:0*
T0*&
_output_shapes
:
encoder/conv2d/ReadVariableOp_1ReadVariableOp&encoder_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0u
encoder/conv2d/Neg_1Neg'encoder/conv2d/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:
encoder/conv2d/add_2AddV2encoder/conv2d/Neg_1:y:0encoder/conv2d/mul_2:z:0*
T0*&
_output_shapes
:[
encoder/conv2d/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
encoder/conv2d/mul_3Mulencoder/conv2d/mul_3/x:output:0encoder/conv2d/add_2:z:0*
T0*&
_output_shapes
:x
encoder/conv2d/StopGradient_1StopGradientencoder/conv2d/mul_3:z:0*
T0*&
_output_shapes
:
encoder/conv2d/ReadVariableOp_2ReadVariableOp&encoder_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
encoder/conv2d/add_3AddV2'encoder/conv2d/ReadVariableOp_2:value:0&encoder/conv2d/StopGradient_1:output:0*
T0*&
_output_shapes
:½
encoder/conv2d/convolutionConv2D$encoder/input_quantization/add_3:z:0encoder/conv2d/add_3:z:0*
T0*&
_output_shapes
:*
paddingSAME*
strides
X
encoder/conv2d/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :X
encoder/conv2d/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : ~
encoder/conv2d/Pow_1Powencoder/conv2d/Pow_1/x:output:0encoder/conv2d/Pow_1/y:output:0*
T0*
_output_shapes
: g
encoder/conv2d/Cast_1Castencoder/conv2d/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 
encoder/conv2d/ReadVariableOp_3ReadVariableOp(encoder_conv2d_readvariableop_3_resource*
_output_shapes
:*
dtype0[
encoder/conv2d/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
encoder/conv2d/mul_4Mul'encoder/conv2d/ReadVariableOp_3:value:0encoder/conv2d/mul_4/y:output:0*
T0*
_output_shapes
:}
encoder/conv2d/truediv_2RealDivencoder/conv2d/mul_4:z:0encoder/conv2d/Cast_1:y:0*
T0*
_output_shapes
:^
encoder/conv2d/Neg_2Negencoder/conv2d/truediv_2:z:0*
T0*
_output_shapes
:b
encoder/conv2d/Round_1Roundencoder/conv2d/truediv_2:z:0*
T0*
_output_shapes
:x
encoder/conv2d/add_4AddV2encoder/conv2d/Neg_2:y:0encoder/conv2d/Round_1:y:0*
T0*
_output_shapes
:l
encoder/conv2d/StopGradient_2StopGradientencoder/conv2d/add_4:z:0*
T0*
_output_shapes
:
encoder/conv2d/add_5AddV2encoder/conv2d/truediv_2:z:0&encoder/conv2d/StopGradient_2:output:0*
T0*
_output_shapes
:m
(encoder/conv2d/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øA£
&encoder/conv2d/clip_by_value_1/MinimumMinimumencoder/conv2d/add_5:z:01encoder/conv2d/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:e
 encoder/conv2d/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â¥
encoder/conv2d/clip_by_value_1Maximum*encoder/conv2d/clip_by_value_1/Minimum:z:0)encoder/conv2d/clip_by_value_1/y:output:0*
T0*
_output_shapes
:
encoder/conv2d/mul_5Mulencoder/conv2d/Cast_1:y:0"encoder/conv2d/clip_by_value_1:z:0*
T0*
_output_shapes
:_
encoder/conv2d/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
encoder/conv2d/truediv_3RealDivencoder/conv2d/mul_5:z:0#encoder/conv2d/truediv_3/y:output:0*
T0*
_output_shapes
:[
encoder/conv2d/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
encoder/conv2d/mul_6Mulencoder/conv2d/mul_6/x:output:0encoder/conv2d/truediv_3:z:0*
T0*
_output_shapes
:
encoder/conv2d/ReadVariableOp_4ReadVariableOp(encoder_conv2d_readvariableop_3_resource*
_output_shapes
:*
dtype0i
encoder/conv2d/Neg_3Neg'encoder/conv2d/ReadVariableOp_4:value:0*
T0*
_output_shapes
:v
encoder/conv2d/add_6AddV2encoder/conv2d/Neg_3:y:0encoder/conv2d/mul_6:z:0*
T0*
_output_shapes
:[
encoder/conv2d/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?{
encoder/conv2d/mul_7Mulencoder/conv2d/mul_7/x:output:0encoder/conv2d/add_6:z:0*
T0*
_output_shapes
:l
encoder/conv2d/StopGradient_3StopGradientencoder/conv2d/mul_7:z:0*
T0*
_output_shapes
:
encoder/conv2d/ReadVariableOp_5ReadVariableOp(encoder_conv2d_readvariableop_3_resource*
_output_shapes
:*
dtype0
encoder/conv2d/add_7AddV2'encoder/conv2d/ReadVariableOp_5:value:0&encoder/conv2d/StopGradient_3:output:0*
T0*
_output_shapes
:
encoder/conv2d/BiasAddBiasAdd#encoder/conv2d/convolution:output:0encoder/conv2d/add_7:z:0*
T0*&
_output_shapes
:S
encoder/act/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :S
encoder/act/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :o
encoder/act/PowPowencoder/act/Pow/x:output:0encoder/act/Pow/y:output:0*
T0*
_output_shapes
: ]
encoder/act/CastCastencoder/act/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: V
encoder/act/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C
encoder/act/mulMulencoder/conv2d/BiasAdd:output:0encoder/act/mul/y:output:0*
T0*&
_output_shapes
:z
encoder/act/truedivRealDivencoder/act/mul:z:0encoder/act/Cast:y:0*
T0*&
_output_shapes
:`
encoder/act/NegNegencoder/act/truediv:z:0*
T0*&
_output_shapes
:d
encoder/act/RoundRoundencoder/act/truediv:z:0*
T0*&
_output_shapes
:u
encoder/act/addAddV2encoder/act/Neg:y:0encoder/act/Round:y:0*
T0*&
_output_shapes
:n
encoder/act/StopGradientStopGradientencoder/act/add:z:0*
T0*&
_output_shapes
:
encoder/act/add_1AddV2encoder/act/truediv:z:0!encoder/act/StopGradient:output:0*
T0*&
_output_shapes
:h
#encoder/act/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB¢
!encoder/act/clip_by_value/MinimumMinimumencoder/act/add_1:z:0,encoder/act/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:`
encoder/act/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã¢
encoder/act/clip_by_valueMaximum%encoder/act/clip_by_value/Minimum:z:0$encoder/act/clip_by_value/y:output:0*
T0*&
_output_shapes
:~
encoder/act/mul_1Mulencoder/act/Cast:y:0encoder/act/clip_by_value:z:0*
T0*&
_output_shapes
:\
encoder/act/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C
encoder/act/truediv_1RealDivencoder/act/mul_1:z:0 encoder/act/truediv_1/y:output:0*
T0*&
_output_shapes
:X
encoder/act/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
encoder/act/mul_2Mulencoder/act/mul_2/x:output:0encoder/act/truediv_1:z:0*
T0*&
_output_shapes
:j
encoder/act/Neg_1Negencoder/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:y
encoder/act/add_2AddV2encoder/act/Neg_1:y:0encoder/act/mul_2:z:0*
T0*&
_output_shapes
:X
encoder/act/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
encoder/act/mul_3Mulencoder/act/mul_3/x:output:0encoder/act/add_2:z:0*
T0*&
_output_shapes
:r
encoder/act/StopGradient_1StopGradientencoder/act/mul_3:z:0*
T0*&
_output_shapes
:
encoder/act/add_3AddV2encoder/conv2d/BiasAdd:output:0#encoder/act/StopGradient_1:output:0*
T0*&
_output_shapes
:
6encoder/tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
8encoder/tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   
8encoder/tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
0encoder/tf.__operators__.getitem_1/strided_sliceStridedSliceinput_1?encoder/tf.__operators__.getitem_1/strided_slice/stack:output:0Aencoder/tf.__operators__.getitem_1/strided_slice/stack_1:output:0Aencoder/tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskf
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
encoder/flatten/ReshapeReshapeencoder/act/add_3:z:0encoder/flatten/Const:output:0*
T0*
_output_shapes
:	
$encoder/tf.compat.v1.squeeze/SqueezeSqueeze9encoder/tf.__operators__.getitem_1/strided_slice:output:0*
T0*
_output_shapes
:U
encoder/dense/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :U
encoder/dense/Pow/yConst*
_output_shapes
: *
dtype0*
value	B : u
encoder/dense/PowPowencoder/dense/Pow/x:output:0encoder/dense/Pow/y:output:0*
T0*
_output_shapes
: a
encoder/dense/CastCastencoder/dense/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: 
encoder/dense/ReadVariableOpReadVariableOp%encoder_dense_readvariableop_resource*
_output_shapes
:	*
dtype0X
encoder/dense/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
encoder/dense/mulMul$encoder/dense/ReadVariableOp:value:0encoder/dense/mul/y:output:0*
T0*
_output_shapes
:	y
encoder/dense/truedivRealDivencoder/dense/mul:z:0encoder/dense/Cast:y:0*
T0*
_output_shapes
:	]
encoder/dense/NegNegencoder/dense/truediv:z:0*
T0*
_output_shapes
:	a
encoder/dense/RoundRoundencoder/dense/truediv:z:0*
T0*
_output_shapes
:	t
encoder/dense/addAddV2encoder/dense/Neg:y:0encoder/dense/Round:y:0*
T0*
_output_shapes
:	k
encoder/dense/StopGradientStopGradientencoder/dense/add:z:0*
T0*
_output_shapes
:	
encoder/dense/add_1AddV2encoder/dense/truediv:z:0#encoder/dense/StopGradient:output:0*
T0*
_output_shapes
:	j
%encoder/dense/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øA¡
#encoder/dense/clip_by_value/MinimumMinimumencoder/dense/add_1:z:0.encoder/dense/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:	b
encoder/dense/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â¡
encoder/dense/clip_by_valueMaximum'encoder/dense/clip_by_value/Minimum:z:0&encoder/dense/clip_by_value/y:output:0*
T0*
_output_shapes
:	}
encoder/dense/mul_1Mulencoder/dense/Cast:y:0encoder/dense/clip_by_value:z:0*
T0*
_output_shapes
:	^
encoder/dense/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
encoder/dense/truediv_1RealDivencoder/dense/mul_1:z:0"encoder/dense/truediv_1/y:output:0*
T0*
_output_shapes
:	Z
encoder/dense/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
encoder/dense/mul_2Mulencoder/dense/mul_2/x:output:0encoder/dense/truediv_1:z:0*
T0*
_output_shapes
:	
encoder/dense/ReadVariableOp_1ReadVariableOp%encoder_dense_readvariableop_resource*
_output_shapes
:	*
dtype0l
encoder/dense/Neg_1Neg&encoder/dense/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	x
encoder/dense/add_2AddV2encoder/dense/Neg_1:y:0encoder/dense/mul_2:z:0*
T0*
_output_shapes
:	Z
encoder/dense/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
encoder/dense/mul_3Mulencoder/dense/mul_3/x:output:0encoder/dense/add_2:z:0*
T0*
_output_shapes
:	o
encoder/dense/StopGradient_1StopGradientencoder/dense/mul_3:z:0*
T0*
_output_shapes
:	
encoder/dense/ReadVariableOp_2ReadVariableOp%encoder_dense_readvariableop_resource*
_output_shapes
:	*
dtype0
encoder/dense/add_3AddV2&encoder/dense/ReadVariableOp_2:value:0%encoder/dense/StopGradient_1:output:0*
T0*
_output_shapes
:	
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0encoder/dense/add_3:z:0*
T0*
_output_shapes

:W
encoder/dense/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :W
encoder/dense/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B : {
encoder/dense/Pow_1Powencoder/dense/Pow_1/x:output:0encoder/dense/Pow_1/y:output:0*
T0*
_output_shapes
: e
encoder/dense/Cast_1Castencoder/dense/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: 
encoder/dense/ReadVariableOp_3ReadVariableOp'encoder_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0Z
encoder/dense/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
encoder/dense/mul_4Mul&encoder/dense/ReadVariableOp_3:value:0encoder/dense/mul_4/y:output:0*
T0*
_output_shapes
:z
encoder/dense/truediv_2RealDivencoder/dense/mul_4:z:0encoder/dense/Cast_1:y:0*
T0*
_output_shapes
:\
encoder/dense/Neg_2Negencoder/dense/truediv_2:z:0*
T0*
_output_shapes
:`
encoder/dense/Round_1Roundencoder/dense/truediv_2:z:0*
T0*
_output_shapes
:u
encoder/dense/add_4AddV2encoder/dense/Neg_2:y:0encoder/dense/Round_1:y:0*
T0*
_output_shapes
:j
encoder/dense/StopGradient_2StopGradientencoder/dense/add_4:z:0*
T0*
_output_shapes
:
encoder/dense/add_5AddV2encoder/dense/truediv_2:z:0%encoder/dense/StopGradient_2:output:0*
T0*
_output_shapes
:l
'encoder/dense/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  øA 
%encoder/dense/clip_by_value_1/MinimumMinimumencoder/dense/add_5:z:00encoder/dense/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:d
encoder/dense/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Â¢
encoder/dense/clip_by_value_1Maximum)encoder/dense/clip_by_value_1/Minimum:z:0(encoder/dense/clip_by_value_1/y:output:0*
T0*
_output_shapes
:|
encoder/dense/mul_5Mulencoder/dense/Cast_1:y:0!encoder/dense/clip_by_value_1:z:0*
T0*
_output_shapes
:^
encoder/dense/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   B
encoder/dense/truediv_3RealDivencoder/dense/mul_5:z:0"encoder/dense/truediv_3/y:output:0*
T0*
_output_shapes
:Z
encoder/dense/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?|
encoder/dense/mul_6Mulencoder/dense/mul_6/x:output:0encoder/dense/truediv_3:z:0*
T0*
_output_shapes
:
encoder/dense/ReadVariableOp_4ReadVariableOp'encoder_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0g
encoder/dense/Neg_3Neg&encoder/dense/ReadVariableOp_4:value:0*
T0*
_output_shapes
:s
encoder/dense/add_6AddV2encoder/dense/Neg_3:y:0encoder/dense/mul_6:z:0*
T0*
_output_shapes
:Z
encoder/dense/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
encoder/dense/mul_7Mulencoder/dense/mul_7/x:output:0encoder/dense/add_6:z:0*
T0*
_output_shapes
:j
encoder/dense/StopGradient_3StopGradientencoder/dense/mul_7:z:0*
T0*
_output_shapes
:
encoder/dense/ReadVariableOp_5ReadVariableOp'encoder_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0
encoder/dense/add_7AddV2&encoder/dense/ReadVariableOp_5:value:0%encoder/dense/StopGradient_3:output:0*
T0*
_output_shapes
:
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0encoder/dense/add_7:z:0*
T0*
_output_shapes

:c
!encoder/latent_quantization/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :c
!encoder/latent_quantization/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :
encoder/latent_quantization/PowPow*encoder/latent_quantization/Pow/x:output:0*encoder/latent_quantization/Pow/y:output:0*
T0*
_output_shapes
: }
 encoder/latent_quantization/CastCast#encoder/latent_quantization/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
!encoder/latent_quantization/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
encoder/latent_quantization/mulMulencoder/dense/BiasAdd:output:0*encoder/latent_quantization/mul/y:output:0*
T0*
_output_shapes

:¢
#encoder/latent_quantization/truedivRealDiv#encoder/latent_quantization/mul:z:0$encoder/latent_quantization/Cast:y:0*
T0*
_output_shapes

:x
encoder/latent_quantization/NegNeg'encoder/latent_quantization/truediv:z:0*
T0*
_output_shapes

:|
!encoder/latent_quantization/RoundRound'encoder/latent_quantization/truediv:z:0*
T0*
_output_shapes

:
encoder/latent_quantization/addAddV2#encoder/latent_quantization/Neg:y:0%encoder/latent_quantization/Round:y:0*
T0*
_output_shapes

:
(encoder/latent_quantization/StopGradientStopGradient#encoder/latent_quantization/add:z:0*
T0*
_output_shapes

:¯
!encoder/latent_quantization/add_1AddV2'encoder/latent_quantization/truediv:z:01encoder/latent_quantization/StopGradient:output:0*
T0*
_output_shapes

:x
3encoder/latent_quantization/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CÊ
1encoder/latent_quantization/clip_by_value/MinimumMinimum%encoder/latent_quantization/add_1:z:0<encoder/latent_quantization/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:p
+encoder/latent_quantization/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÃÊ
)encoder/latent_quantization/clip_by_valueMaximum5encoder/latent_quantization/clip_by_value/Minimum:z:04encoder/latent_quantization/clip_by_value/y:output:0*
T0*
_output_shapes

:¦
!encoder/latent_quantization/mul_1Mul$encoder/latent_quantization/Cast:y:0-encoder/latent_quantization/clip_by_value:z:0*
T0*
_output_shapes

:l
'encoder/latent_quantization/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C²
%encoder/latent_quantization/truediv_1RealDiv%encoder/latent_quantization/mul_1:z:00encoder/latent_quantization/truediv_1/y:output:0*
T0*
_output_shapes

:h
#encoder/latent_quantization/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ª
!encoder/latent_quantization/mul_2Mul,encoder/latent_quantization/mul_2/x:output:0)encoder/latent_quantization/truediv_1:z:0*
T0*
_output_shapes

:q
!encoder/latent_quantization/Neg_1Negencoder/dense/BiasAdd:output:0*
T0*
_output_shapes

:¡
!encoder/latent_quantization/add_2AddV2%encoder/latent_quantization/Neg_1:y:0%encoder/latent_quantization/mul_2:z:0*
T0*
_output_shapes

:h
#encoder/latent_quantization/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¦
!encoder/latent_quantization/mul_3Mul,encoder/latent_quantization/mul_3/x:output:0%encoder/latent_quantization/add_2:z:0*
T0*
_output_shapes

:
*encoder/latent_quantization/StopGradient_1StopGradient%encoder/latent_quantization/mul_3:z:0*
T0*
_output_shapes

:¨
!encoder/latent_quantization/add_3AddV2encoder/dense/BiasAdd:output:03encoder/latent_quantization/StopGradient_1:output:0*
T0*
_output_shapes

:g
%encoder/tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ·
!encoder/tf.expand_dims/ExpandDims
ExpandDims-encoder/tf.compat.v1.squeeze/Squeeze:output:0.encoder/tf.expand_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:a
encoder/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Õ
encoder/concatenate/concatConcatV2%encoder/latent_quantization/add_3:z:0*encoder/tf.expand_dims/ExpandDims:output:0(encoder/concatenate/concat/axis:output:0*
N*
T0*
_output_shapes

:i
IdentityIdentity#encoder/concatenate/concat:output:0^NoOp*
T0*
_output_shapes

:°
NoOpNoOp^encoder/conv2d/ReadVariableOp ^encoder/conv2d/ReadVariableOp_1 ^encoder/conv2d/ReadVariableOp_2 ^encoder/conv2d/ReadVariableOp_3 ^encoder/conv2d/ReadVariableOp_4 ^encoder/conv2d/ReadVariableOp_5^encoder/dense/ReadVariableOp^encoder/dense/ReadVariableOp_1^encoder/dense/ReadVariableOp_2^encoder/dense/ReadVariableOp_3^encoder/dense/ReadVariableOp_4^encoder/dense/ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:	: : : : 2B
encoder/conv2d/ReadVariableOp_1encoder/conv2d/ReadVariableOp_12B
encoder/conv2d/ReadVariableOp_2encoder/conv2d/ReadVariableOp_22B
encoder/conv2d/ReadVariableOp_3encoder/conv2d/ReadVariableOp_32B
encoder/conv2d/ReadVariableOp_4encoder/conv2d/ReadVariableOp_42B
encoder/conv2d/ReadVariableOp_5encoder/conv2d/ReadVariableOp_52>
encoder/conv2d/ReadVariableOpencoder/conv2d/ReadVariableOp2@
encoder/dense/ReadVariableOp_1encoder/dense/ReadVariableOp_12@
encoder/dense/ReadVariableOp_2encoder/dense/ReadVariableOp_22@
encoder/dense/ReadVariableOp_3encoder/dense/ReadVariableOp_32@
encoder/dense/ReadVariableOp_4encoder/dense/ReadVariableOp_42@
encoder/dense/ReadVariableOp_5encoder/dense/ReadVariableOp_52<
encoder/dense/ReadVariableOpencoder/dense/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
&
_output_shapes
:	
!
_user_specified_name	input_1
Â

$__inference_dense_layer_call_fn_1583

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1251f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1579:$ 

_user_specified_name1577:G C

_output_shapes
:	
 
_user_specified_nameinputs
é

%__inference_conv2d_layer_call_fn_1459

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1134n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1455:$ 

_user_specified_name1453:N J
&
_output_shapes
:
 
_user_specified_nameinputs
ÿ
Ó
&__inference_encoder_layer_call_fn_1356
input_1!
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_1330f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1352:$ 

_user_specified_name1350:$ 

_user_specified_name1348:$ 

_user_specified_name1346:O K
&
_output_shapes
:	
!
_user_specified_name	input_1
²
Y
=__inference_act_layer_call_and_return_conditional_losses_1563

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CS
mulMulinputsmul/y:output:0*
T0*&
_output_shapes
:V
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:Z
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:E
Neg_1Neginputs*
T0*&
_output_shapes
:U
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:`
add_3AddV2inputsStopGradient_1:output:0*
T0*&
_output_shapes
:P
IdentityIdentity	add_3:z:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
Ò-

__inference__traced_save_1746
file_prefix>
$read_disablecopyonread_conv2d_kernel:2
$read_1_disablecopyonread_conv2d_bias:8
%read_2_disablecopyonread_dense_kernel:	1
#read_3_disablecopyonread_dense_bias:
savev2_const

identity_9¢MergeV2Checkpoints¢Read/DisableCopyOnRead¢Read/ReadVariableOp¢Read_1/DisableCopyOnRead¢Read_1/ReadVariableOp¢Read_2/DisableCopyOnRead¢Read_2/ReadVariableOp¢Read_3/DisableCopyOnRead¢Read_3/ReadVariableOpw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_conv2d_kernel"/device:CPU:0*
_output_shapes
 ¨
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_conv2d_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_conv2d_bias"/device:CPU:0*
_output_shapes
  
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_conv2d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:y
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 ¦
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_dense_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	w
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_dense_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:æ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHw
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B °
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes	
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_8Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_9IdentityIdentity_8:output:0^NoOp*
T0*
_output_shapes
: ÿ
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp*
_output_shapes
 "!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:+'
%
_user_specified_nameconv2d/bias:-)
'
_user_specified_nameconv2d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
*

A__inference_encoder_layer_call_and_return_conditional_losses_1300
input_1%
conv2d_1135:
conv2d_1137:

dense_1252:	

dense_1254:
identity¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      È
&tf.__operators__.getitem/strided_sliceStridedSliceinput_15tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_maskõ
"input_quantization/PartitionedCallPartitionedCall/tf.__operators__.getitem/strided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_input_quantization_layer_call_and_return_conditional_losses_1065
conv2d/StatefulPartitionedCallStatefulPartitionedCall+input_quantization/PartitionedCall:output:0conv2d_1135conv2d_1137*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1134Ï
act/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_act_layer_call_and_return_conditional_losses_1170
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ä
(tf.__operators__.getitem_1/strided_sliceStridedSliceinput_17tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskÅ
flatten/PartitionedCallPartitionedCallact/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1181
tf.compat.v1.squeeze/SqueezeSqueeze1tf.__operators__.getitem_1/strided_slice:output:0*
T0*
_output_shapes
:ò
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_1252
dense_1254*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1251æ
#latent_quantization/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_latent_quantization_layer_call_and_return_conditional_losses_1287_
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
tf.expand_dims/ExpandDims
ExpandDims%tf.compat.v1.squeeze/Squeeze:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:
concatenate/PartitionedCallPartitionedCall,latent_quantization/PartitionedCall:output:0"tf.expand_dims/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1297j
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*
_output_shapes

:c
NoOpNoOp^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:	: : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:$ 

_user_specified_name1254:$ 

_user_specified_name1252:$ 

_user_specified_name1137:$ 

_user_specified_name1135:O K
&
_output_shapes
:	
!
_user_specified_name	input_1
²
Y
=__inference_act_layer_call_and_return_conditional_losses_1170

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CS
mulMulinputsmul/y:output:0*
T0*&
_output_shapes
:V
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:Z
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:E
Neg_1Neginputs*
T0*&
_output_shapes
:U
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:`
add_3AddV2inputsStopGradient_1:output:0*
T0*&
_output_shapes
:P
IdentityIdentity	add_3:z:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
À
á
 __inference__traced_restore_1767
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:2
assignvariableop_2_dense_kernel:	+
assignvariableop_3_dense_bias:

identity_5¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3é
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHz
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B ·
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ¬

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_5IdentityIdentity_4:output:0^NoOp_1*
T0*
_output_shapes
: v
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
_output_shapes
 "!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32$
AssignVariableOpAssignVariableOp:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:+'
%
_user_specified_nameconv2d/bias:-)
'
_user_specified_nameconv2d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ù
Ï
"__inference_signature_wrapper_1414
input_1!
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_1028f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1410:$ 

_user_specified_name1408:$ 

_user_specified_name1406:$ 

_user_specified_name1404:O K
&
_output_shapes
:	
!
_user_specified_name	input_1
ÿ
Ó
&__inference_encoder_layer_call_fn_1343
input_1!
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_encoder_layer_call_and_return_conditional_losses_1300f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1339:$ 

_user_specified_name1337:$ 

_user_specified_name1335:$ 

_user_specified_name1333:O K
&
_output_shapes
:	
!
_user_specified_name	input_1
¬
M
1__inference_input_quantization_layer_call_fn_1419

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_input_quantization_layer_call_and_return_conditional_losses_1065_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs
*

A__inference_encoder_layer_call_and_return_conditional_losses_1330
input_1%
conv2d_1308:
conv2d_1310:

dense_1320:	

dense_1322:
identity¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCall}
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      È
&tf.__operators__.getitem/strided_sliceStridedSliceinput_15tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:*

begin_mask*
end_maskõ
"input_quantization/PartitionedCallPartitionedCall/tf.__operators__.getitem/strided_slice:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_input_quantization_layer_call_and_return_conditional_losses_1065
conv2d/StatefulPartitionedCallStatefulPartitionedCall+input_quantization/PartitionedCall:output:0conv2d_1308conv2d_1310*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1134Ï
act/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_act_layer_call_and_return_conditional_losses_1170
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ä
(tf.__operators__.getitem_1/strided_sliceStridedSliceinput_17tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*"
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskÅ
flatten/PartitionedCallPartitionedCallact/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1181
tf.compat.v1.squeeze/SqueezeSqueeze1tf.__operators__.getitem_1/strided_slice:output:0*
T0*
_output_shapes
:ò
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dense_1320
dense_1322*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_1251æ
#latent_quantization/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_latent_quantization_layer_call_and_return_conditional_losses_1287_
tf.expand_dims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
tf.expand_dims/ExpandDims
ExpandDims%tf.compat.v1.squeeze/Squeeze:output:0&tf.expand_dims/ExpandDims/dim:output:0*
T0*
_output_shapes

:
concatenate/PartitionedCallPartitionedCall,latent_quantization/PartitionedCall:output:0"tf.expand_dims/ExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_1297j
IdentityIdentity$concatenate/PartitionedCall:output:0^NoOp*
T0*
_output_shapes

:c
NoOpNoOp^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:	: : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:$ 

_user_specified_name1322:$ 

_user_specified_name1320:$ 

_user_specified_name1310:$ 

_user_specified_name1308:O K
&
_output_shapes
:	
!
_user_specified_name	input_1
Á
h
L__inference_input_quantization_layer_call_and_return_conditional_losses_1065

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CS
mulMulinputsmul/y:output:0*
T0*&
_output_shapes
:V
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:H
NegNegtruediv:z:0*
T0*&
_output_shapes
:L
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:Q
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:V
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:c
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:Z
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:E
Neg_1Neginputs*
T0*&
_output_shapes
:U
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:Z
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:`
add_3AddV2inputsStopGradient_1:output:0*
T0*&
_output_shapes
:P
IdentityIdentity	add_3:z:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
::N J
&
_output_shapes
:
 
_user_specified_nameinputs"ÊL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¤
serving_default
:
input_1/
serving_default_input_1:0	6
concatenate'
StatefulPartitionedCall:0tensorflow/serving/predict:è
¦
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
	layer-8

layer-9
layer-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
Ä
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

activation
	quantizer"
_tf_keras_layer
Ó
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#kernel_quantizer
$bias_quantizer
#kernel_quantizer_internal
$bias_quantizer_internal
%
quantizers

&kernel
'bias
 (_jit_compiled_convolution_op"
_tf_keras_layer
Ä
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/
activation
/	quantizer"
_tf_keras_layer
¥
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
(
6	keras_api"
_tf_keras_layer
±
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=kernel_quantizer
>bias_quantizer
=kernel_quantizer_internal
>bias_quantizer_internal
?
quantizers

@kernel
Abias"
_tf_keras_layer
(
B	keras_api"
_tf_keras_layer
Ä
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I
activation
I	quantizer"
_tf_keras_layer
(
J	keras_api"
_tf_keras_layer
¥
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
<
&0
'1
@2
A3"
trackable_list_wrapper
<
&0
'1
@2
A3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¿
Vtrace_0
Wtrace_12
&__inference_encoder_layer_call_fn_1343
&__inference_encoder_layer_call_fn_1356µ
®²ª
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zVtrace_0zWtrace_1
õ
Xtrace_0
Ytrace_12¾
A__inference_encoder_layer_call_and_return_conditional_losses_1300
A__inference_encoder_layer_call_and_return_conditional_losses_1330µ
®²ª
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zXtrace_0zYtrace_1
ÊBÇ
__inference__wrapped_model_1028input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Zserving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ë
`trace_02Î
1__inference_input_quantization_layer_call_fn_1419
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z`trace_0

atrace_02é
L__inference_input_quantization_layer_call_and_return_conditional_losses_1450
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zatrace_0
"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
ß
gtrace_02Â
%__inference_conv2d_layer_call_fn_1459
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zgtrace_0
ú
htrace_02Ý
@__inference_conv2d_layer_call_and_return_conditional_losses_1527
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zhtrace_0
"
_generic_user_object
"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
':%2conv2d/kernel
:2conv2d/bias
ª2§¤
²
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Ü
ntrace_02¿
"__inference_act_layer_call_fn_1532
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zntrace_0
÷
otrace_02Ú
=__inference_act_layer_call_and_return_conditional_losses_1563
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
à
utrace_02Ã
&__inference_flatten_layer_call_fn_1568
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zutrace_0
û
vtrace_02Þ
A__inference_flatten_layer_call_and_return_conditional_losses_1574
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zvtrace_0
"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
Þ
|trace_02Á
$__inference_dense_layer_call_fn_1583
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z|trace_0
ù
}trace_02Ü
?__inference_dense_layer_call_and_return_conditional_losses_1651
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z}trace_0
"
_generic_user_object
"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
:	2dense/kernel
:2
dense/bias
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
2__inference_latent_quantization_layer_call_fn_1656
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
M__inference_latent_quantization_layer_call_and_return_conditional_losses_1687
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
æ
trace_02Ç
*__inference_concatenate_layer_call_fn_1693
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02â
E__inference_concatenate_layer_call_and_return_conditional_losses_1700
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
îBë
&__inference_encoder_layer_call_fn_1343input_1"µ
®²ª
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
îBë
&__inference_encoder_layer_call_fn_1356input_1"µ
®²ª
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
A__inference_encoder_layer_call_and_return_conditional_losses_1300input_1"µ
®²ª
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
A__inference_encoder_layer_call_and_return_conditional_losses_1330input_1"µ
®²ª
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults¢
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÉBÆ
"__inference_signature_wrapper_1414input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÛBØ
1__inference_input_quantization_layer_call_fn_1419inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
L__inference_input_quantization_layer_call_and_return_conditional_losses_1450inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÏBÌ
%__inference_conv2d_layer_call_fn_1459inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
êBç
@__inference_conv2d_layer_call_and_return_conditional_losses_1527inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÌBÉ
"__inference_act_layer_call_fn_1532inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
çBä
=__inference_act_layer_call_and_return_conditional_losses_1563inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÐBÍ
&__inference_flatten_layer_call_fn_1568inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ëBè
A__inference_flatten_layer_call_and_return_conditional_losses_1574inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÎBË
$__inference_dense_layer_call_fn_1583inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
éBæ
?__inference_dense_layer_call_and_return_conditional_losses_1651inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÜBÙ
2__inference_latent_quantization_layer_call_fn_1656inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
M__inference_latent_quantization_layer_call_and_return_conditional_losses_1687inputs"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
àBÝ
*__inference_concatenate_layer_call_fn_1693inputs_0inputs_1"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
E__inference_concatenate_layer_call_and_return_conditional_losses_1700inputs_0inputs_1"
²
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
__inference__wrapped_model_1028i&'@A/¢,
%¢"
 
input_1	
ª "0ª-
+
concatenate
concatenate
=__inference_act_layer_call_and_return_conditional_losses_1563].¢+
$¢!

inputs
ª "+¢(
!
tensor_0
 x
"__inference_act_layer_call_fn_1532R.¢+
$¢!

inputs
ª " 
unknown¸
E__inference_concatenate_layer_call_and_return_conditional_losses_1700oH¢E
>¢;
96

inputs_0

inputs_1
ª "#¢ 

tensor_0
 
*__inference_concatenate_layer_call_fn_1693dH¢E
>¢;
96

inputs_0

inputs_1
ª "
unknown¥
@__inference_conv2d_layer_call_and_return_conditional_losses_1527a&'.¢+
$¢!

inputs
ª "+¢(
!
tensor_0
 
%__inference_conv2d_layer_call_fn_1459V&'.¢+
$¢!

inputs
ª " 
unknown
?__inference_dense_layer_call_and_return_conditional_losses_1651R@A'¢$
¢

inputs	
ª "#¢ 

tensor_0
 o
$__inference_dense_layer_call_fn_1583G@A'¢$
¢

inputs	
ª "
unknown©
A__inference_encoder_layer_call_and_return_conditional_losses_1300d&'@A7¢4
-¢*
 
input_1	
p

 
ª "#¢ 

tensor_0
 ©
A__inference_encoder_layer_call_and_return_conditional_losses_1330d&'@A7¢4
-¢*
 
input_1	
p 

 
ª "#¢ 

tensor_0
 
&__inference_encoder_layer_call_fn_1343Y&'@A7¢4
-¢*
 
input_1	
p

 
ª "
unknown
&__inference_encoder_layer_call_fn_1356Y&'@A7¢4
-¢*
 
input_1	
p 

 
ª "
unknown
A__inference_flatten_layer_call_and_return_conditional_losses_1574V.¢+
$¢!

inputs
ª "$¢!

tensor_0	
 u
&__inference_flatten_layer_call_fn_1568K.¢+
$¢!

inputs
ª "
unknown	­
L__inference_input_quantization_layer_call_and_return_conditional_losses_1450].¢+
$¢!

inputs
ª "+¢(
!
tensor_0
 
1__inference_input_quantization_layer_call_fn_1419R.¢+
$¢!

inputs
ª " 
unknown
M__inference_latent_quantization_layer_call_and_return_conditional_losses_1687M&¢#
¢

inputs
ª "#¢ 

tensor_0
 x
2__inference_latent_quantization_layer_call_fn_1656B&¢#
¢

inputs
ª "
unknown
"__inference_signature_wrapper_1414t&'@A:¢7
¢ 
0ª-
+
input_1 
input_1	"0ª-
+
concatenate
concatenate