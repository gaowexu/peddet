import * as cdk from '@aws-cdk/core';
import * as lambda from '@aws-cdk/aws-lambda';
import * as agw from '@aws-cdk/aws-apigateway';
import * as iam from '@aws-cdk/aws-iam';
import * as s3 from '@aws-cdk/aws-s3';
import * as sagemaker from '@aws-cdk/aws-sagemaker';


export class IpcAiSaasStack extends cdk.Stack {
  constructor(scope: cdk.Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    this.templateOptions.description = '(SO8016) - IP Camera AI SaaS Service Stack. Template version v1.0.0';

    const imageName = new cdk.CfnParameter(this, 'imageName', {
        description: 'Specify your image name in ECR',
        type: 'String',
    });

    const saveRequestEvents = new cdk.CfnParameter(this, 'saveRequestEvents', {
        description: 'Whether to save all request images and corresponding response for close-loop improvement',
        type: 'String',
        default: 'No',
        allowedValues: [
            'Yes',
            'No',
        ]
    });


    /**
     * Default Deployment Machine Type
     */
    const deployInstanceType = 'ml.g4dn.xlarge';


    /**
     * S3 Bucket Provision
     */
    const events = new s3.Bucket(
        this,
        'events',
        {
            removalPolicy: cdk.RemovalPolicy.DESTROY,
            autoDeleteObjects: true,
        }
    );


    /**
     * Sagemaker Model/Endpoint Configuration/Endpoint Provision
     */
    const sagemakerExecuteRole = new iam.Role(
        this,
        'sagemakerExecuteRole',
        {
            assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
            managedPolicies: [
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3FullAccess'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryFullAccess'),
                iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchLogsFullAccess'),
            ]
        }
    );

    const imageUrl = `${cdk.Aws.ACCOUNT_ID}.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com/${imageName.valueAsString}:latest`;


    // create model
    const sagemakerEndpointModel = new sagemaker.CfnModel(
        this,
        'sagemakerEndpointModel',
        {
            executionRoleArn: sagemakerExecuteRole.roleArn,
            containers: [
                {
                    image: imageUrl.toString(),
                    mode: 'SingleModel',
                }
            ],
        }
    );

    // create endpoint configuration
    const sagemakerEndpointConfig = new sagemaker.CfnEndpointConfig(
        this,
        'sagemakerEndpointConfig',
        {
            productionVariants: [{
                initialInstanceCount: 1,
                initialVariantWeight: 1,
                instanceType: deployInstanceType,
                modelName: sagemakerEndpointModel.attrModelName,
                variantName: 'AllTraffic',
            }]
        }
    );

    // create endpoint
    const sagemakerEndpoint = new sagemaker.CfnEndpoint(
        this,
        'sagemakerEndpoint',
        {
            endpointConfigName: sagemakerEndpointConfig.attrEndpointConfigName
        }
    );


    /**
     * Lambda Function & API Gateway Provision
     */
    const lambdaAccessPolicy = new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
            "sagemaker:InvokeEndpoint",
            "s3:GetObject",
            "s3:PutObject",
        ],
        resources: ["*"]
    });
    lambdaAccessPolicy.addAllResources();

    // lambda function provision
    const ipcSaaSHandler = new lambda.Function(
        this,
        'ipcSaaSHandler',
        {
            code: new lambda.AssetCode( 'lambda'),
            handler: 'main.handler',
            runtime: lambda.Runtime.PYTHON_3_8,
            environment: {
                SAGEMAKER_ENDPOINT_NAME: sagemakerEndpoint.attrEndpointName,
                EVENTS_S3_BUCKET_NAME: events.bucketName,
                REQUEST_EVENTS_SNAPSHOT_ENABLED: `${saveRequestEvents.valueAsString}`,
            },
            timeout: cdk.Duration.minutes(10),
            memorySize: 512,
        }
    );
    ipcSaaSHandler.addToRolePolicy(lambdaAccessPolicy);

    // api gateway provision
    const apiRouter = new agw.RestApi(
        this,
        'apiRouter',
        {
            endpointConfiguration: {
                types: [agw.EndpointType.REGIONAL]
            },
            defaultCorsPreflightOptions: {
                allowOrigins: agw.Cors.ALL_ORIGINS,
                allowMethods: agw.Cors.ALL_METHODS
            }
        }
    );
    apiRouter.root.addResource('inference').addMethod('POST', new agw.LambdaIntegration(ipcSaaSHandler));

  }
}
