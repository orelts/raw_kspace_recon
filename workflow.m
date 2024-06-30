files = ["epi_se.seq",
"meas_MID00146_FID04783_Anatomy_for_us_t2_space_sag_cs4_iso.dat"
"meas_MID00148_FID04785_t1_se_sag.dat"
"meas_MID00150_FID04787_t1_se_sag_fov_phase_50%.dat"
"meas_MID00151_FID04788_t1_se_sag_Phase_Resolusion_50%.dat"
"meas_MID00159_FID04796_t1_se_sag_Phase_Resolusion_50%.dat"
"meas_MID00169_FID04806_pulseq.dat"];


twix = mapVBVD('/home/orel/Downloads/kspace/meas_MID00148_FID04785_t1_se_sag.dat');

twix = twix{end} ; 

twix.image.flagDoAverage = true;
twix.image.flagRemoveOS  = true;
%%

% raw = twix_obj.image() ;
% ImgPerChannel = raw ;
% for DimIdx = [1,3]
%     ImgPerChannel = ifftshift(fft(fftshift(ImgPerChannel, DimIdx), [], DimIdx), DimIdx) ;
% end
% Img = squeeze(sqrt(sum(abs(ImgPerChannel).^2, 2))) ;

% The final matrix size of our image (single precision is enough):

os  = 2; % oversampling factor
img = zeros([twix.image.NCol/os, twix.image.NLin, twix.image.NSli], 'single');

for sli = 1:twix.image.NSli
    % read in the data for slice 'sli'
    data = twix.image(:,:,:,1,sli);
    
    % fft in col and lin dimension:
    fft_dims = [1 3];
    for f = fft_dims
        data = fftshift(ifft(ifftshift(data,f),[],f),f);
    end
    
    % sum-of-square coil combination:
    img(:,:,sli) = squeeze(sqrt(sum(abs(data).^2,2)));
end

%%

s = sliceViewer(img,DisplayRange=[0, 0.4 * max(img(:))])
% imagesc(flipud(img(:,:,5)), [0, 0.5*max(img(:))]), colormap gray, axis image off;
