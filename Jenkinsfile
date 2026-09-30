pipeline {
    agent any

    environment {
        // the app EC2 instance's ID (not its IP) -- find it in the EC2 console
        APP_INSTANCE_ID = 'i-0d7764e52d3ab8c83'
        APP_URL         = 'https://18-191-50-6.sslip.io'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Set up environment') {
            steps {
                sh '''#!/bin/bash
                echo "Setting up required dependencies"
                python3 -m venv .venv
                . .venv/bin/activate
                pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
                pip install --no-cache-dir -r requirements.txt
                pip install --no-cache-dir pytest
                '''
            }
        }

        stage('Run user input test') {
            steps {
                sh '''#!/bin/bash
                . .venv/bin/activate
                python -m pytest test_user_input.py
                '''
            }
        }

        stage('Run deeper analysis test') {
            steps {
                sh '''#!/bin/bash
                . .venv/bin/activate
                python -m pytest test_resume_analysis.py
                '''
            }
        }

        stage('Deploy') {
            steps {
                sh '''#!/bin/bash
                echo "Deploying the application"
                COMMAND_ID=$(aws ssm send-command \
                    --instance-ids "$APP_INSTANCE_ID" \
                    --document-name "AWS-RunShellScript" \
                    --parameters 'commands=["git config --system --add safe.directory /home/ubuntu/jobpostproject","cd /home/ubuntu/jobpostproject && git pull && docker compose up -d --build"]' \
                    --query "Command.CommandId" --output text)

                aws ssm wait command-executed \
                    --command-id "$COMMAND_ID" \
                    --instance-id "$APP_INSTANCE_ID"
                '''
            }
        }

        stage('Evaluate') {
            steps {
                sh '''#!/bin/bash
                curl -sf --retry 5 --retry-delay 3 --retry-all-errors "$APP_URL" > /dev/null
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check console logs.'
        }
        always {
            echo 'Jenkins job finished.'
        }
    }
}
