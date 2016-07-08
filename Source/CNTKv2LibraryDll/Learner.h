//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace CNTK 
{
    namespace Learners
    {
        // A collection of additional options that are applicable for all standard learners. 
        struct AdditionalLearningOptions
        {
            double l1RegularizationWeight = 0.0;
            double l2RegularizationWeight = 0.0;
            double gaussianNoiseInjectionStdDev = 0.0;
            bool gradientClippingWithTruncation = false;
            double gradientClippingThresholdPerSample = 0.0;
            _Internal::_SimpleMap<Variable, double> learningRateMultipliers;

            void SetLearningRateMultipliers(const std::unordered_map<Variable, double>& multipliers)
            {
               learningRateMultipliers = _Internal::_SimpleMap<Variable, double>::CreateSimpleMap(multipliers);
            }
        };

        // An abstract base class at the root of the standard learners hierarchy
        // It implements most of the learner functionality, except for the actual update function,
        // and adds a few pre-/postprocessing methods (which are invoked before and after the update).
        class LearnerBase : public Learner
        {
        public:

            virtual Dictionary GetCheckpointState() const override;

            virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) override;

            // TODO: move learning rate and momentum scheduling and adjustment functionality 
            // inside the learner and drop these setters.
            void SetLearningRate(double value) { m_learningRatePerSample = value; }

            void CNTK_API SetAdditionalOptions(const AdditionalLearningOptions& additionalOptions);

             // TODO: should this be called ResetMomentum?
             // needed for BlockMomemtumSGD to reset SGD momentum after aggregation.
            void CNTK_API ResetSmoothedGradients();

        protected:
            LearnerBase(const _Internal::_SimpleSet<Variable>& parameters, 
                        const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

            virtual bool Update(const _Internal::_SimpleMap<Variable, ValuePtr>& parameterValues,
                                const _Internal::_SimpleMap<Variable, const ValuePtr>& gradientValues,
                                size_t trainingSampleCount) override final;

            virtual void Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const = 0;

            double ParameterDependentLearningRate(const Variable& parameter) const
            {
                return m_learningRatePerSample * m_additionalOptions.learningRateMultipliers[parameter];
            }


            virtual std::wstring LearnerType() = 0;

            double m_learningRatePerSample;

            AdditionalLearningOptions m_additionalOptions;

            size_t m_sampleCount;

            _Internal::_SimpleSet<Variable> m_parameters;

            _Internal::_SimpleMap<Variable, ValuePtr> m_smoothedGradientValues;

            // The following four static protected methods expose private methods of NDArrayView class
            // (which declares LearnerBase as friend class), so that they are available to subclasses.
            template <typename ElementType>
            static std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>> GetMatrix(const NDArrayViewPtr arrayView);

            template <typename ElementType>
            static std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>> GetWritableMatrix(NDArrayViewPtr arrayView);

            template <typename ElementType>
            static const Microsoft::MSR::CNTK::TensorView<ElementType>* GetTensorView(const NDArrayViewPtr arrayView);

            template <typename ElementType>
            static Microsoft::MSR::CNTK::TensorView<ElementType>* GetWritableTensorView(NDArrayViewPtr arrayView);

            template <typename ElementType>
            void ClipGradient(Microsoft::MSR::CNTK::Matrix<ElementType>& gradient, size_t actualMBSize) const;

            // Performs additional preprocessing before calling the update method 
            // (gradient clipping and L2 regularization depending on the additional learning parameters).
            template <typename ElementType>
            bool PreProcess(const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t actualMBSize) const;

            // Performs additional postprocessing after the update method has been executed
            // (noise injection and L1 regularization specified by the additional learning parameters).
            template <typename ElementType>
            bool PostProcess(const Variable& parameter, const ValuePtr& gradientValue, 
                             const ValuePtr& parameterValue, size_t actualMBSize) const;
        private:
            // TODO: make these functions friends of NDViewArray and move to Utils?
            static bool HasNan(const ValuePtr& value, const char* name);
            static void Print(const ValuePtr& value, const char* msg);
        };

         // A base call for all learners that use NormalGrad update. 
        class SGDLearnerBase : public LearnerBase
        {
        protected:

            SGDLearnerBase(const _Internal::_SimpleSet<Variable>& parameters,
                       const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

            virtual void Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const override;

            template <typename ElementType>
            bool Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                        const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const;

            virtual std::wstring LearnerType() = 0;

            double m_momentumPerSample;
            bool m_useNesterovAcceleration;
        };


        // Vanilla gradient descent optimization algorithm 
        class SGDLearner : public SGDLearnerBase
        {
        public:

            SGDLearner(const _Internal::_SimpleSet<Variable>& parameters,
                       const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice())
                       : SGDLearnerBase(parameters, device)
            {
            }

        protected:

            virtual std::wstring LearnerType() override { return L"SGD Learner"; }
        };

        // SGD optimization with momentum. 
        class MomentumSGDLearner : public SGDLearnerBase
        {
        public:

            MomentumSGDLearner(const _Internal::_SimpleSet<Variable>& parameters,
                               const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice())
                               : SGDLearnerBase(parameters, device)
            {
            }
            
            void SetMomentum(double value) { m_momentumPerSample = value; }

        protected:

            virtual std::wstring LearnerType() override { return L"Momentum SGD Learner"; }
        };

        // Nesterov's accelerated SGDLearnerBase descent. 
        class NAGLearner : public SGDLearnerBase
        {
        public:

            NAGLearner(const _Internal::_SimpleSet<Variable>& parameters,
                               const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice())
                               : SGDLearnerBase(parameters, device)
            {
                m_useNesterovAcceleration = true;
            }

        protected:

            virtual std::wstring LearnerType() override { return L"NAG Learner"; }
        };

        class AdaGradLearner : public LearnerBase
        {
        public:

            AdaGradLearner(const _Internal::_SimpleSet<Variable>& parameters, bool needAveMultiplier,
                           const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

        protected:
            bool m_needAveMultiplier;

            virtual void Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const override;

            template <typename ElementType>
            bool Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                        const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const;

            virtual std::wstring LearnerType() override { return L"AdaGrad Learner"; }
        };

        class FSAdaGradLearner : public MomentumSGDLearner
        {
        public:

            FSAdaGradLearner(const _Internal::_SimpleSet<Variable>& parameters,
                             const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

        protected:

            virtual void Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const override;

            template <typename ElementType>
            bool Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                        const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const;

            virtual std::wstring LearnerType() override { return L"FSAdaGrad Learner"; }
        };

        class RMSPropLearner : public LearnerBase
        {
        public:

            RMSPropLearner(const _Internal::_SimpleSet<Variable>& parameters,
                           double gamma, double inc, double dec, double max, double min, bool needAveMultiplier,
                           const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

        protected:

            double m_gamma;
            double m_inc;
            double m_dec;
            double m_max;
            double m_min;
            bool m_needAveMultiplier;

            virtual void Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const override;

            template <typename ElementType>
            bool Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                        const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const;

            virtual std::wstring LearnerType() override { return L"RMSProp Learner"; }
        };
    }
}