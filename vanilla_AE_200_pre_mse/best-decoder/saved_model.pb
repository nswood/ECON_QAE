Á
À
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
 "serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628×¡
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
:d*
dtype0*
shape:d
ë
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasdense/kernel
dense/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_9376219

NoOpNoOp
&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Å%
value»%B¸% B±%
Ì
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
­
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

activation
	quantizer* 
¾
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
kernel_quantizer
bias_quantizer
kernel_quantizer_internal
bias_quantizer_internal

quantizers

 kernel
!bias
 "_jit_compiled_convolution_op*
­
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)
activation
)	quantizer* 

*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 

0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6kernel_quantizer
7bias_quantizer
6kernel_quantizer_internal
7bias_quantizer_internal
8
quantizers

9kernel
:bias*
­
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A
activation
A	quantizer* 
 
 0
!1
92
:3*
 
 0
!1
92
:3*
* 
°
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Gtrace_0
Htrace_1* 

Itrace_0
Jtrace_1* 
* 

Kserving_default* 
* 
* 
* 

Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Qtrace_0* 

Rtrace_0* 
* 

 0
!1*

 0
!1*
* 

Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
* 
* 

0
1* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

_trace_0* 

`trace_0* 
* 
* 
* 
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

ftrace_0* 

gtrace_0* 

90
:1*

90
:1*
* 

hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
* 
* 

60
71* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

ttrace_0* 

utrace_0* 
* 
* 
5
0
1
2
3
4
5
6*
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
Ù
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_9376538
Ô
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_9376559·ê
µ
\
@__inference_act_layer_call_and_return_conditional_losses_9376008

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
:dV
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:dH
NegNegtruediv:z:0*
T0*&
_output_shapes
:dL
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:dQ
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:dV
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:dc
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:d\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:dZ
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:dP
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:dL
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:dE
Neg_1Neginputs*
T0*&
_output_shapes
:dU
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:dL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:dZ
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:d`
add_3AddV2inputsStopGradient_1:output:0*
T0*&
_output_shapes
:dP
IdentityIdentity	add_3:z:0*
T0*&
_output_shapes
:d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:d:N J
&
_output_shapes
:d
 
_user_specified_nameinputs
ë
Ò
%__inference_signature_wrapper_9376219
input_1!
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_9375870f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9376215:'#
!
_user_specified_name	9376213:'#
!
_user_specified_name	9376211:'#
!
_user_specified_name	9376209:O K
&
_output_shapes
:d
!
_user_specified_name	input_1
þ3
®
C__inference_conv2d_layer_call_and_return_conditional_losses_9376332

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
:d*
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
:d^
IdentityIdentityBiasAdd:output:0^NoOp*
T0*&
_output_shapes
:d
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:d: : 2$
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
:d
 
_user_specified_nameinputs

E
)__inference_flatten_layer_call_fn_9376373

inputs
identityª
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9376015X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:d:N J
&
_output_shapes
:d
 
_user_specified_nameinputs
Ã
ä
#__inference__traced_restore_9376559
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
Î

'__inference_dense_layer_call_fn_9376388

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9376084f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	d: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9376384:'#
!
_user_specified_name	9376382:G C

_output_shapes
:	d
 
_user_specified_nameinputs
é
¬
D__inference_encoder_layer_call_and_return_conditional_losses_9376141
input_1(
conv2d_9376127:
conv2d_9376129: 
dense_9376134:	
dense_9376136:
identity¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCallÐ
"input_quantization/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_input_quantization_layer_call_and_return_conditional_losses_9375903
conv2d/StatefulPartitionedCallStatefulPartitionedCall+input_quantization/PartitionedCall:output:0conv2d_9376127conv2d_9376129*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9375972Ò
act/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_act_layer_call_and_return_conditional_losses_9376008È
flatten/PartitionedCallPartitionedCallact/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9376015û
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_9376134dense_9376136*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9376084é
#latent_quantization/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_latent_quantization_layer_call_and_return_conditional_losses_9376120r
IdentityIdentity,latent_quantization/PartitionedCall:output:0^NoOp*
T0*
_output_shapes

:dc
NoOpNoOp^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:d: : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:'#
!
_user_specified_name	9376136:'#
!
_user_specified_name	9376134:'#
!
_user_specified_name	9376129:'#
!
_user_specified_name	9376127:O K
&
_output_shapes
:d
!
_user_specified_name	input_1
Õ-

 __inference__traced_save_9376538
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
Ä
k
O__inference_input_quantization_layer_call_and_return_conditional_losses_9375903

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
:dV
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:dH
NegNegtruediv:z:0*
T0*&
_output_shapes
:dL
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:dQ
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:dV
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:dc
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:d\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:dZ
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:dP
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:dL
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:dE
Neg_1Neginputs*
T0*&
_output_shapes
:dU
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:dL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:dZ
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:d`
add_3AddV2inputsStopGradient_1:output:0*
T0*&
_output_shapes
:dP
IdentityIdentity	add_3:z:0*
T0*&
_output_shapes
:d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:d:N J
&
_output_shapes
:d
 
_user_specified_nameinputs
2
¦
B__inference_dense_layer_call_and_return_conditional_losses_9376084

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

:dI
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

:dV
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:d
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	d: : 2$
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
:	d
 
_user_specified_nameinputs

Q
5__inference_latent_quantization_layer_call_fn_9376461

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_latent_quantization_layer_call_and_return_conditional_losses_9376120W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

:d:F B

_output_shapes

:d
 
_user_specified_nameinputs
¦ß
»
"__inference__wrapped_model_9375870
input_1@
&encoder_conv2d_readvariableop_resource:6
(encoder_conv2d_readvariableop_3_resource:8
%encoder_dense_readvariableop_resource:	5
'encoder_dense_readvariableop_3_resource:
identity¢encoder/conv2d/ReadVariableOp¢encoder/conv2d/ReadVariableOp_1¢encoder/conv2d/ReadVariableOp_2¢encoder/conv2d/ReadVariableOp_3¢encoder/conv2d/ReadVariableOp_4¢encoder/conv2d/ReadVariableOp_5¢encoder/dense/ReadVariableOp¢encoder/dense/ReadVariableOp_1¢encoder/dense/ReadVariableOp_2¢encoder/dense/ReadVariableOp_3¢encoder/dense/ReadVariableOp_4¢encoder/dense/ReadVariableOp_5b
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
 *   C
encoder/input_quantization/mulMulinput_1)encoder/input_quantization/mul/y:output:0*
T0*&
_output_shapes
:d§
"encoder/input_quantization/truedivRealDiv"encoder/input_quantization/mul:z:0#encoder/input_quantization/Cast:y:0*
T0*&
_output_shapes
:d~
encoder/input_quantization/NegNeg&encoder/input_quantization/truediv:z:0*
T0*&
_output_shapes
:d
 encoder/input_quantization/RoundRound&encoder/input_quantization/truediv:z:0*
T0*&
_output_shapes
:d¢
encoder/input_quantization/addAddV2"encoder/input_quantization/Neg:y:0$encoder/input_quantization/Round:y:0*
T0*&
_output_shapes
:d
'encoder/input_quantization/StopGradientStopGradient"encoder/input_quantization/add:z:0*
T0*&
_output_shapes
:d´
 encoder/input_quantization/add_1AddV2&encoder/input_quantization/truediv:z:00encoder/input_quantization/StopGradient:output:0*
T0*&
_output_shapes
:dw
2encoder/input_quantization/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þBÏ
0encoder/input_quantization/clip_by_value/MinimumMinimum$encoder/input_quantization/add_1:z:0;encoder/input_quantization/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:do
*encoder/input_quantization/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ÃÏ
(encoder/input_quantization/clip_by_valueMaximum4encoder/input_quantization/clip_by_value/Minimum:z:03encoder/input_quantization/clip_by_value/y:output:0*
T0*&
_output_shapes
:d«
 encoder/input_quantization/mul_1Mul#encoder/input_quantization/Cast:y:0,encoder/input_quantization/clip_by_value:z:0*
T0*&
_output_shapes
:dk
&encoder/input_quantization/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C·
$encoder/input_quantization/truediv_1RealDiv$encoder/input_quantization/mul_1:z:0/encoder/input_quantization/truediv_1/y:output:0*
T0*&
_output_shapes
:dg
"encoder/input_quantization/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
 encoder/input_quantization/mul_2Mul+encoder/input_quantization/mul_2/x:output:0(encoder/input_quantization/truediv_1:z:0*
T0*&
_output_shapes
:da
 encoder/input_quantization/Neg_1Neginput_1*
T0*&
_output_shapes
:d¦
 encoder/input_quantization/add_2AddV2$encoder/input_quantization/Neg_1:y:0$encoder/input_quantization/mul_2:z:0*
T0*&
_output_shapes
:dg
"encoder/input_quantization/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?«
 encoder/input_quantization/mul_3Mul+encoder/input_quantization/mul_3/x:output:0$encoder/input_quantization/add_2:z:0*
T0*&
_output_shapes
:d
)encoder/input_quantization/StopGradient_1StopGradient$encoder/input_quantization/mul_3:z:0*
T0*&
_output_shapes
:d
 encoder/input_quantization/add_3AddV2input_12encoder/input_quantization/StopGradient_1:output:0*
T0*&
_output_shapes
:dV
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
:d*
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
:dS
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
:dz
encoder/act/truedivRealDivencoder/act/mul:z:0encoder/act/Cast:y:0*
T0*&
_output_shapes
:d`
encoder/act/NegNegencoder/act/truediv:z:0*
T0*&
_output_shapes
:dd
encoder/act/RoundRoundencoder/act/truediv:z:0*
T0*&
_output_shapes
:du
encoder/act/addAddV2encoder/act/Neg:y:0encoder/act/Round:y:0*
T0*&
_output_shapes
:dn
encoder/act/StopGradientStopGradientencoder/act/add:z:0*
T0*&
_output_shapes
:d
encoder/act/add_1AddV2encoder/act/truediv:z:0!encoder/act/StopGradient:output:0*
T0*&
_output_shapes
:dh
#encoder/act/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB¢
!encoder/act/clip_by_value/MinimumMinimumencoder/act/add_1:z:0,encoder/act/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:d`
encoder/act/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã¢
encoder/act/clip_by_valueMaximum%encoder/act/clip_by_value/Minimum:z:0$encoder/act/clip_by_value/y:output:0*
T0*&
_output_shapes
:d~
encoder/act/mul_1Mulencoder/act/Cast:y:0encoder/act/clip_by_value:z:0*
T0*&
_output_shapes
:d\
encoder/act/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C
encoder/act/truediv_1RealDivencoder/act/mul_1:z:0 encoder/act/truediv_1/y:output:0*
T0*&
_output_shapes
:dX
encoder/act/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
encoder/act/mul_2Mulencoder/act/mul_2/x:output:0encoder/act/truediv_1:z:0*
T0*&
_output_shapes
:dj
encoder/act/Neg_1Negencoder/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:dy
encoder/act/add_2AddV2encoder/act/Neg_1:y:0encoder/act/mul_2:z:0*
T0*&
_output_shapes
:dX
encoder/act/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?~
encoder/act/mul_3Mulencoder/act/mul_3/x:output:0encoder/act/add_2:z:0*
T0*&
_output_shapes
:dr
encoder/act/StopGradient_1StopGradientencoder/act/mul_3:z:0*
T0*&
_output_shapes
:d
encoder/act/add_3AddV2encoder/conv2d/BiasAdd:output:0#encoder/act/StopGradient_1:output:0*
T0*&
_output_shapes
:df
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
encoder/flatten/ReshapeReshapeencoder/act/add_3:z:0encoder/flatten/Const:output:0*
T0*
_output_shapes
:	dU
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

:dW
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

:dc
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

:d¢
#encoder/latent_quantization/truedivRealDiv#encoder/latent_quantization/mul:z:0$encoder/latent_quantization/Cast:y:0*
T0*
_output_shapes

:dx
encoder/latent_quantization/NegNeg'encoder/latent_quantization/truediv:z:0*
T0*
_output_shapes

:d|
!encoder/latent_quantization/RoundRound'encoder/latent_quantization/truediv:z:0*
T0*
_output_shapes

:d
encoder/latent_quantization/addAddV2#encoder/latent_quantization/Neg:y:0%encoder/latent_quantization/Round:y:0*
T0*
_output_shapes

:d
(encoder/latent_quantization/StopGradientStopGradient#encoder/latent_quantization/add:z:0*
T0*
_output_shapes

:d¯
!encoder/latent_quantization/add_1AddV2'encoder/latent_quantization/truediv:z:01encoder/latent_quantization/StopGradient:output:0*
T0*
_output_shapes

:dx
3encoder/latent_quantization/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  CÊ
1encoder/latent_quantization/clip_by_value/MinimumMinimum%encoder/latent_quantization/add_1:z:0<encoder/latent_quantization/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:dp
+encoder/latent_quantization/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÃÊ
)encoder/latent_quantization/clip_by_valueMaximum5encoder/latent_quantization/clip_by_value/Minimum:z:04encoder/latent_quantization/clip_by_value/y:output:0*
T0*
_output_shapes

:d¦
!encoder/latent_quantization/mul_1Mul$encoder/latent_quantization/Cast:y:0-encoder/latent_quantization/clip_by_value:z:0*
T0*
_output_shapes

:dl
'encoder/latent_quantization/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C²
%encoder/latent_quantization/truediv_1RealDiv%encoder/latent_quantization/mul_1:z:00encoder/latent_quantization/truediv_1/y:output:0*
T0*
_output_shapes

:dh
#encoder/latent_quantization/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ª
!encoder/latent_quantization/mul_2Mul,encoder/latent_quantization/mul_2/x:output:0)encoder/latent_quantization/truediv_1:z:0*
T0*
_output_shapes

:dq
!encoder/latent_quantization/Neg_1Negencoder/dense/BiasAdd:output:0*
T0*
_output_shapes

:d¡
!encoder/latent_quantization/add_2AddV2%encoder/latent_quantization/Neg_1:y:0%encoder/latent_quantization/mul_2:z:0*
T0*
_output_shapes

:dh
#encoder/latent_quantization/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¦
!encoder/latent_quantization/mul_3Mul,encoder/latent_quantization/mul_3/x:output:0%encoder/latent_quantization/add_2:z:0*
T0*
_output_shapes

:d
*encoder/latent_quantization/StopGradient_1StopGradient%encoder/latent_quantization/mul_3:z:0*
T0*
_output_shapes

:d¨
!encoder/latent_quantization/add_3AddV2encoder/dense/BiasAdd:output:03encoder/latent_quantization/StopGradient_1:output:0*
T0*
_output_shapes

:dk
IdentityIdentity%encoder/latent_quantization/add_3:z:0^NoOp*
T0*
_output_shapes

:d°
NoOpNoOp^encoder/conv2d/ReadVariableOp ^encoder/conv2d/ReadVariableOp_1 ^encoder/conv2d/ReadVariableOp_2 ^encoder/conv2d/ReadVariableOp_3 ^encoder/conv2d/ReadVariableOp_4 ^encoder/conv2d/ReadVariableOp_5^encoder/dense/ReadVariableOp^encoder/dense/ReadVariableOp_1^encoder/dense/ReadVariableOp_2^encoder/dense/ReadVariableOp_3^encoder/dense/ReadVariableOp_4^encoder/dense/ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:d: : : : 2B
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
:d
!
_user_specified_name	input_1
²
P
4__inference_input_quantization_layer_call_fn_9376224

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_input_quantization_layer_call_and_return_conditional_losses_9375903_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:d:N J
&
_output_shapes
:d
 
_user_specified_nameinputs
¥
l
P__inference_latent_quantization_layer_call_and_return_conditional_losses_9376120

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

:dN
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:d@
NegNegtruediv:z:0*
T0*
_output_shapes

:dD
RoundRoundtruediv:z:0*
T0*
_output_shapes

:dI
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:dN
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:d[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:d\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ãv
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:dR
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:dP
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:dL
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:d=
Neg_1Neginputs*
T0*
_output_shapes

:dM
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:dL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:dR
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:dX
add_3AddV2inputsStopGradient_1:output:0*
T0*
_output_shapes

:dH
IdentityIdentity	add_3:z:0*
T0*
_output_shapes

:d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

:d:F B

_output_shapes

:d
 
_user_specified_nameinputs
Ä
k
O__inference_input_quantization_layer_call_and_return_conditional_losses_9376255

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
:dV
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:dH
NegNegtruediv:z:0*
T0*&
_output_shapes
:dL
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:dQ
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:dV
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:dc
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:d\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:dZ
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:dP
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:dL
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:dE
Neg_1Neginputs*
T0*&
_output_shapes
:dU
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:dL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:dZ
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:d`
add_3AddV2inputsStopGradient_1:output:0*
T0*&
_output_shapes
:dP
IdentityIdentity	add_3:z:0*
T0*&
_output_shapes
:d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:d:N J
&
_output_shapes
:d
 
_user_specified_nameinputs
¢
`
D__inference_flatten_layer_call_and_return_conditional_losses_9376015

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
:	dP
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:d:N J
&
_output_shapes
:d
 
_user_specified_nameinputs

A
%__inference_act_layer_call_fn_9376337

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_act_layer_call_and_return_conditional_losses_9376008_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:d:N J
&
_output_shapes
:d
 
_user_specified_nameinputs
þ3
®
C__inference_conv2d_layer_call_and_return_conditional_losses_9375972

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
:d*
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
:d^
IdentityIdentityBiasAdd:output:0^NoOp*
T0*&
_output_shapes
:d
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:d: : 2$
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
:d
 
_user_specified_nameinputs
	
Ö
)__inference_encoder_layer_call_fn_9376154
input_1!
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_9376123f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9376150:'#
!
_user_specified_name	9376148:'#
!
_user_specified_name	9376146:'#
!
_user_specified_name	9376144:O K
&
_output_shapes
:d
!
_user_specified_name	input_1
¥
l
P__inference_latent_quantization_layer_call_and_return_conditional_losses_9376492

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

:dN
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:d@
NegNegtruediv:z:0*
T0*
_output_shapes

:dD
RoundRoundtruediv:z:0*
T0*
_output_shapes

:dI
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:dN
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:d[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:d\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ãv
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:dR
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:dP
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:dL
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:d=
Neg_1Neginputs*
T0*
_output_shapes

:dM
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:dL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:dR
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:dX
add_3AddV2inputsStopGradient_1:output:0*
T0*
_output_shapes

:dH
IdentityIdentity	add_3:z:0*
T0*
_output_shapes

:d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

:d:F B

_output_shapes

:d
 
_user_specified_nameinputs
µ
\
@__inference_act_layer_call_and_return_conditional_losses_9376368

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
:dV
truedivRealDivmul:z:0Cast:y:0*
T0*&
_output_shapes
:dH
NegNegtruediv:z:0*
T0*&
_output_shapes
:dL
RoundRoundtruediv:z:0*
T0*&
_output_shapes
:dQ
addAddV2Neg:y:0	Round:y:0*
T0*&
_output_shapes
:dV
StopGradientStopGradientadd:z:0*
T0*&
_output_shapes
:dc
add_1AddV2truediv:z:0StopGradient:output:0*
T0*&
_output_shapes
:d\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  þB~
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Ã~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:dZ
mul_1MulCast:y:0clip_by_value:z:0*
T0*&
_output_shapes
:dP
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cf
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*&
_output_shapes
:dL
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*&
_output_shapes
:dE
Neg_1Neginputs*
T0*&
_output_shapes
:dU
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*&
_output_shapes
:dL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*&
_output_shapes
:dZ
StopGradient_1StopGradient	mul_3:z:0*
T0*&
_output_shapes
:d`
add_3AddV2inputsStopGradient_1:output:0*
T0*&
_output_shapes
:dP
IdentityIdentity	add_3:z:0*
T0*&
_output_shapes
:d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:d:N J
&
_output_shapes
:d
 
_user_specified_nameinputs
¢
`
D__inference_flatten_layer_call_and_return_conditional_losses_9376379

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
:	dP
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:d:N J
&
_output_shapes
:d
 
_user_specified_nameinputs
é
¬
D__inference_encoder_layer_call_and_return_conditional_losses_9376123
input_1(
conv2d_9375973:
conv2d_9375975: 
dense_9376085:	
dense_9376087:
identity¢conv2d/StatefulPartitionedCall¢dense/StatefulPartitionedCallÐ
"input_quantization/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_input_quantization_layer_call_and_return_conditional_losses_9375903
conv2d/StatefulPartitionedCallStatefulPartitionedCall+input_quantization/PartitionedCall:output:0conv2d_9375973conv2d_9375975*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9375972Ò
act/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_act_layer_call_and_return_conditional_losses_9376008È
flatten/PartitionedCallPartitionedCallact/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9376015û
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_9376085dense_9376087*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9376084é
#latent_quantization/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_latent_quantization_layer_call_and_return_conditional_losses_9376120r
IdentityIdentity,latent_quantization/PartitionedCall:output:0^NoOp*
T0*
_output_shapes

:dc
NoOpNoOp^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:d: : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:'#
!
_user_specified_name	9376087:'#
!
_user_specified_name	9376085:'#
!
_user_specified_name	9375975:'#
!
_user_specified_name	9375973:O K
&
_output_shapes
:d
!
_user_specified_name	input_1
õ

(__inference_conv2d_layer_call_fn_9376264

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_9375972n
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*&
_output_shapes
:d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:d: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9376260:'#
!
_user_specified_name	9376258:N J
&
_output_shapes
:d
 
_user_specified_nameinputs
2
¦
B__inference_dense_layer_call_and_return_conditional_losses_9376456

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

:dI
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

:dV
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:d
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	d: : 2$
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
:	d
 
_user_specified_nameinputs
	
Ö
)__inference_encoder_layer_call_fn_9376167
input_1!
unknown:
	unknown_0:
	unknown_1:	
	unknown_2:
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_9376141f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:d<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9376163:'#
!
_user_specified_name	9376161:'#
!
_user_specified_name	9376159:'#
!
_user_specified_name	9376157:O K
&
_output_shapes
:d
!
_user_specified_name	input_1"ÊL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
:
input_1/
serving_default_input_1:0d>
latent_quantization'
StatefulPartitionedCall:0dtensorflow/serving/predict:
ã
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ä
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

activation
	quantizer"
_tf_keras_layer
Ó
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
kernel_quantizer
bias_quantizer
kernel_quantizer_internal
bias_quantizer_internal

quantizers

 kernel
!bias
 "_jit_compiled_convolution_op"
_tf_keras_layer
Ä
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)
activation
)	quantizer"
_tf_keras_layer
¥
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
±
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6kernel_quantizer
7bias_quantizer
6kernel_quantizer_internal
7bias_quantizer_internal
8
quantizers

9kernel
:bias"
_tf_keras_layer
Ä
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A
activation
A	quantizer"
_tf_keras_layer
<
 0
!1
92
:3"
trackable_list_wrapper
<
 0
!1
92
:3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Å
Gtrace_0
Htrace_12
)__inference_encoder_layer_call_fn_9376154
)__inference_encoder_layer_call_fn_9376167µ
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
 zGtrace_0zHtrace_1
û
Itrace_0
Jtrace_12Ä
D__inference_encoder_layer_call_and_return_conditional_losses_9376123
D__inference_encoder_layer_call_and_return_conditional_losses_9376141µ
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
 zItrace_0zJtrace_1
ÍBÊ
"__inference__wrapped_model_9375870input_1"
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
Kserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
Qtrace_02Ñ
4__inference_input_quantization_layer_call_fn_9376224
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
 zQtrace_0

Rtrace_02ì
O__inference_input_quantization_layer_call_and_return_conditional_losses_9376255
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
 zRtrace_0
"
_generic_user_object
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
â
Xtrace_02Å
(__inference_conv2d_layer_call_fn_9376264
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
 zXtrace_0
ý
Ytrace_02à
C__inference_conv2d_layer_call_and_return_conditional_losses_9376332
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
 zYtrace_0
"
_generic_user_object
"
_generic_user_object
.
0
1"
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
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ß
_trace_02Â
%__inference_act_layer_call_fn_9376337
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
 z_trace_0
ú
`trace_02Ý
@__inference_act_layer_call_and_return_conditional_losses_9376368
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
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ã
ftrace_02Æ
)__inference_flatten_layer_call_fn_9376373
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
 zftrace_0
þ
gtrace_02á
D__inference_flatten_layer_call_and_return_conditional_losses_9376379
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
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
á
mtrace_02Ä
'__inference_dense_layer_call_fn_9376388
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
 zmtrace_0
ü
ntrace_02ß
B__inference_dense_layer_call_and_return_conditional_losses_9376456
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
"
_generic_user_object
"
_generic_user_object
.
60
71"
trackable_list_wrapper
:	2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
ï
ttrace_02Ò
5__inference_latent_quantization_layer_call_fn_9376461
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
 zttrace_0

utrace_02í
P__inference_latent_quantization_layer_call_and_return_conditional_losses_9376492
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
"
_generic_user_object
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ñBî
)__inference_encoder_layer_call_fn_9376154input_1"µ
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
ñBî
)__inference_encoder_layer_call_fn_9376167input_1"µ
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
B
D__inference_encoder_layer_call_and_return_conditional_losses_9376123input_1"µ
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
B
D__inference_encoder_layer_call_and_return_conditional_losses_9376141input_1"µ
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
ÌBÉ
%__inference_signature_wrapper_9376219input_1"
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
ÞBÛ
4__inference_input_quantization_layer_call_fn_9376224inputs"
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
ùBö
O__inference_input_quantization_layer_call_and_return_conditional_losses_9376255inputs"
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
ÒBÏ
(__inference_conv2d_layer_call_fn_9376264inputs"
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
íBê
C__inference_conv2d_layer_call_and_return_conditional_losses_9376332inputs"
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
%__inference_act_layer_call_fn_9376337inputs"
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
@__inference_act_layer_call_and_return_conditional_losses_9376368inputs"
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
ÓBÐ
)__inference_flatten_layer_call_fn_9376373inputs"
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
îBë
D__inference_flatten_layer_call_and_return_conditional_losses_9376379inputs"
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
ÑBÎ
'__inference_dense_layer_call_fn_9376388inputs"
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
ìBé
B__inference_dense_layer_call_and_return_conditional_losses_9376456inputs"
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
ßBÜ
5__inference_latent_quantization_layer_call_fn_9376461inputs"
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
úB÷
P__inference_latent_quantization_layer_call_and_return_conditional_losses_9376492inputs"
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
 
"__inference__wrapped_model_9375870y !9:/¢,
%¢"
 
input_1d
ª "@ª=
;
latent_quantization$!
latent_quantizationd¡
@__inference_act_layer_call_and_return_conditional_losses_9376368].¢+
$¢!

inputsd
ª "+¢(
!
tensor_0d
 {
%__inference_act_layer_call_fn_9376337R.¢+
$¢!

inputsd
ª " 
unknownd¨
C__inference_conv2d_layer_call_and_return_conditional_losses_9376332a !.¢+
$¢!

inputsd
ª "+¢(
!
tensor_0d
 
(__inference_conv2d_layer_call_fn_9376264V !.¢+
$¢!

inputsd
ª " 
unknownd
B__inference_dense_layer_call_and_return_conditional_losses_9376456R9:'¢$
¢

inputs	d
ª "#¢ 

tensor_0d
 r
'__inference_dense_layer_call_fn_9376388G9:'¢$
¢

inputs	d
ª "
unknownd¬
D__inference_encoder_layer_call_and_return_conditional_losses_9376123d !9:7¢4
-¢*
 
input_1d
p

 
ª "#¢ 

tensor_0d
 ¬
D__inference_encoder_layer_call_and_return_conditional_losses_9376141d !9:7¢4
-¢*
 
input_1d
p 

 
ª "#¢ 

tensor_0d
 
)__inference_encoder_layer_call_fn_9376154Y !9:7¢4
-¢*
 
input_1d
p

 
ª "
unknownd
)__inference_encoder_layer_call_fn_9376167Y !9:7¢4
-¢*
 
input_1d
p 

 
ª "
unknownd
D__inference_flatten_layer_call_and_return_conditional_losses_9376379V.¢+
$¢!

inputsd
ª "$¢!

tensor_0	d
 x
)__inference_flatten_layer_call_fn_9376373K.¢+
$¢!

inputsd
ª "
unknown	d°
O__inference_input_quantization_layer_call_and_return_conditional_losses_9376255].¢+
$¢!

inputsd
ª "+¢(
!
tensor_0d
 
4__inference_input_quantization_layer_call_fn_9376224R.¢+
$¢!

inputsd
ª " 
unknownd¡
P__inference_latent_quantization_layer_call_and_return_conditional_losses_9376492M&¢#
¢

inputsd
ª "#¢ 

tensor_0d
 {
5__inference_latent_quantization_layer_call_fn_9376461B&¢#
¢

inputsd
ª "
unknownd®
%__inference_signature_wrapper_9376219 !9::¢7
¢ 
0ª-
+
input_1 
input_1d"@ª=
;
latent_quantization$!
latent_quantizationd